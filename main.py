#!/usr/bin/env python3
"""
Cursor-like Code Indexing Client
命令行入口程序
"""

import asyncio
import click
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax
import time
import json

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.client import CodeIndexClient
from src.utils import validate_project_path, get_project_info, format_file_size, format_duration

console = Console()


@click.group()
@click.option('--config-path', type=click.Path(), help='配置文件路径')
@click.option('--storage-path', type=click.Path(), help='存储目录路径')
@click.pass_context
def cli(ctx, config_path, storage_path):
    """Cursor-like 代码索引客户端"""
    ctx.ensure_object(dict)
    
    # 初始化客户端
    config_path = Path(config_path) if config_path else None
    storage_path = Path(storage_path) if storage_path else None
    print(f"config_path",config_path)
    try:
        ctx.obj['client'] = CodeIndexClient(storage_path, config_path)
    except Exception as e:
        console.print(f"[red]初始化客户端失败: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('project_path', type=click.Path(exists=True))
@click.option('--force', is_flag=True, help='强制重建索引')
@click.option('--max-tokens', type=int, default=1000, help='每个代码块的最大token数')
@click.pass_context
def index(ctx, project_path, force, max_tokens):
    """索引项目代码库"""
    client = ctx.obj['client']
    project_path = str(Path(project_path).resolve())
    
    # 验证项目路径
    if not validate_project_path(project_path):
        console.print(f"[red]无效的项目路径: {project_path}[/red]")
        return
    
    # 显示项目信息
    project_info = get_project_info(project_path)
    
    console.print(Panel.fit(
        f"[bold]项目信息[/bold]\n"
        f"名称: {project_info['name']}\n"
        f"路径: {project_path}\n"
        f"文件数: {project_info['file_count']}\n"
        f"代码文件数: {project_info['code_file_count']}\n"
        f"大小: {format_file_size(project_info['size_bytes'])}\n"
        f"语言: {', '.join(project_info['languages']) if project_info['languages'] else '未检测到'}",
        title="项目概览"
    ))
    
    # 设置配置
    if max_tokens != 1000:
        client.config.set('max_tokens_per_chunk', max_tokens)
    
    # 开始索引
    console.print(f"\n[yellow]开始索引项目{'（强制重建）' if force else ''}...[/yellow]")
    
    start_time = time.time()
    
    async def run_index():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("正在索引...", total=None)
                
                result = await client.index_project(project_path, force_rebuild=force)
                
                progress.update(task, description="索引完成!")
                return result
        except Exception as e:
            console.print(f"[red]索引失败: {e}[/red]")
            return None
    
    result = asyncio.run(run_index())
    
    if result:
        duration = time.time() - start_time
        
        # 显示结果
        status_color = "green" if result['status'] == 'completed' else "yellow"
        console.print(f"\n[{status_color}]索引{result['status']}![/{status_color}]")
        
        # 创建结果表格
        table = Table(title="索引结果")
        table.add_column("项目", style="cyan")
        table.add_column("状态", style="magenta")
        table.add_column("文件数", style="green")
        table.add_column("代码块数", style="blue")
        table.add_column("处理文件数", style="yellow")
        table.add_column("耗时", style="red")
        
        table.add_row(
            project_info['name'],
            result['status'],
            str(result['total_files']),
            str(result['total_chunks']),
            str(result.get('changed_files', 0)),
            format_duration(duration)
        )
        
        console.print(table)


@cli.command()
@click.argument('query')
@click.option('--project', type=click.Path(), help='项目路径（可选）')
@click.option('--top-k', type=int, default=10, help='返回结果数量')
@click.option('--language', help='过滤编程语言')
@click.option('--show-code', is_flag=True, help='显示代码内容')
@click.pass_context
def search(ctx, query, project, top_k, language, show_code):
    """搜索代码"""
    client = ctx.obj['client']
    
    if project:
        project = str(Path(project).resolve())
    
    console.print(f"[yellow]搜索查询: {query}[/yellow]")
    
    async def run_search():
        try:
            results = await client.search(
                query=query,
                project_path=project,
                top_k=top_k,
                filter_language=language
            )
            return results
        except Exception as e:
            console.print(f"[red]搜索失败: {e}[/red]")
            return []
    
    results = asyncio.run(run_search())
    
    if not results:
        console.print("[yellow]未找到相关结果[/yellow]")
        return
    
    # 显示搜索结果
    for i, result in enumerate(results, 1):
        similarity_color = "green" if result['similarity'] > 0.8 else "yellow" if result['similarity'] > 0.6 else "red"
        
        console.print(f"\n[bold cyan]{i}. {result['file_path']}[/bold cyan]")
        console.print(f"   相似度: [{similarity_color}]{result['similarity']:.3f}[/{similarity_color}]")
        console.print(f"   语言: {result['language']} | 类型: {result['chunk_type']}")
        console.print(f"   行数: {result['start_line']}-{result['end_line']}")
        
        if show_code:
            # 读取并显示代码
            try:
                if project:
                    file_path = Path(project) / result['file_path']
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    start_line = max(0, result['start_line'] - 1)
                    end_line = min(len(lines), result['end_line'])
                    code_content = ''.join(lines[start_line:end_line])
                    
                    syntax = Syntax(code_content, result['language'], line_numbers=True, 
                                  start_line=result['start_line'])
                    console.print(syntax)
            except Exception as e:
                console.print(f"   [red]无法读取代码: {e}[/red]")


@cli.command()
@click.argument('query')
@click.option('--project', type=click.Path(), help='项目路径（可选）')
@click.option('--max-chunks', type=int, default=5, help='最大上下文块数')
@click.pass_context
def context(ctx, query, project, max_chunks):
    """获取查询相关的代码上下文"""
    client = ctx.obj['client']
    
    if project:
        project = str(Path(project).resolve())
    
    console.print(f"[yellow]获取上下文: {query}[/yellow]")
    
    async def run_context():
        try:
            result = await client.get_context_for_query(
                query=query,
                project_path=project,
                max_context_chunks=max_chunks
            )
            return result
        except Exception as e:
            console.print(f"[red]获取上下文失败: {e}[/red]")
            return None
    
    result = asyncio.run(run_context())
    
    if not result or not result['context']:
        console.print("[yellow]未找到相关上下文[/yellow]")
        return
    
    console.print(f"\n[green]找到 {result['total_chunks']} 个相关代码块:[/green]")
    
    # 显示上下文
    console.print(Panel(
        result['context'],
        title="代码上下文",
        expand=False
    ))


@cli.command()
@click.option('--project', type=click.Path(), help='项目路径')
@click.pass_context
def status(ctx, project):
    """查看项目状态"""
    client = ctx.obj['client']
    
    if project:
        project = str(Path(project).resolve())
        project_status = client.get_project_status(project)
        
        # 显示单个项目状态
        if project_status['status'] == 'not_indexed':
            console.print(f"[red]项目未索引: {project}[/red]")
        else:
            console.print(Panel.fit(
                f"[bold]项目状态[/bold]\n"
                f"名称: {project_status['project_name']}\n"
                f"路径: {project_status['project_path']}\n"
                f"状态: {'[green]已索引[/green]' if project_status['status'] == 'indexed' else '[red]未索引[/red]'}\n"
                f"最后索引: {time.ctime(project_status.get('last_indexed', 0))}\n"
                f"文件数: {project_status.get('total_files', 0)}\n"
                f"代码块数: {project_status.get('total_chunks', 0)}\n"
                f"需要更新: {'[yellow]是[/yellow]' if project_status.get('needs_update') else '[green]否[/green]'}",
                title="项目状态"
            ))
    else:
        # 显示所有项目状态
        projects = client.list_projects()
        
        if not projects:
            console.print("[yellow]没有已索引的项目[/yellow]")
            return
        
        table = Table(title="已索引项目")
        table.add_column("项目名", style="cyan")
        table.add_column("路径", style="blue")
        table.add_column("最后索引", style="green")
        table.add_column("文件数", style="magenta")
        table.add_column("代码块数", style="yellow")
        table.add_column("存储大小", style="red")
        
        for project_data in projects:
            last_indexed = time.ctime(project_data.get('last_indexed', 0))
            storage_size = format_file_size(project_data.get('storage_size_mb', 0) * 1024 * 1024)
            
            table.add_row(
                project_data.get('project_name', 'Unknown'),
                project_data.get('project_path', 'Unknown'),
                last_indexed,
                str(project_data.get('total_files', 0)),
                str(project_data.get('total_chunks', 0)),
                storage_size
            )
        
        console.print(table)


@cli.command()
@click.argument('project_path', type=click.Path(exists=True))
@click.pass_context
def update(ctx, project_path):
    """更新项目索引"""
    client = ctx.obj['client']
    project_path = str(Path(project_path).resolve())
    
    console.print(f"[yellow]更新项目索引: {project_path}[/yellow]")
    
    async def run_update():
        try:
            result = await client.update_project_index(project_path)
            return result
        except Exception as e:
            console.print(f"[red]更新失败: {e}[/red]")
            return None
    
    result = asyncio.run(run_update())
    
    if result:
        console.print(f"[green]更新完成![/green]")
        console.print(f"处理了 {result.get('changed_files', 0)} 个变更文件")


@cli.command()
@click.pass_context
def stats(ctx):
    """显示客户端统计信息"""
    client = ctx.obj['client']
    
    stats = client.get_stats()
    
    # 存储统计
    storage_stats = stats.get('storage', {})
    console.print(Panel.fit(
        f"[bold]存储统计[/bold]\n"
        f"项目数: {storage_stats.get('total_projects', 0)}\n"
        f"总存储: {format_file_size(storage_stats.get('total_storage_mb', 0) * 1024 * 1024)}\n"
        f"项目数据: {format_file_size(storage_stats.get('projects_dir_mb', 0) * 1024 * 1024)}\n"
        f"缓存数据: {format_file_size(storage_stats.get('cache_dir_mb', 0) * 1024 * 1024)}",
        title="存储信息"
    ))
    
    # 配置信息
    config_stats = stats.get('config', {})
    console.print(Panel.fit(
        f"[bold]配置信息[/bold]\n"
        f"嵌入模型: {config_stats.get('embedding_model', 'Unknown')}\n"
        f"最大Token数: {config_stats.get('max_tokens_per_chunk', 'Unknown')}\n"
        f"支持扩展: {', '.join(config_stats.get('supported_extensions', []))}",
        title="配置信息"
    ))


@cli.command()
@click.argument('project_path', type=click.Path())
@click.option('--confirm', is_flag=True, help='确认删除')
@click.pass_context
def delete(ctx, project_path, confirm):
    """删除项目索引"""
    client = ctx.obj['client']
    project_path = str(Path(project_path).resolve())
    
    if not confirm:
        if not click.confirm(f'确定要删除项目 "{project_path}" 的索引数据吗？'):
            return
    
    success = client.delete_project_index(project_path)
    
    if success:
        console.print(f"[green]已删除项目索引: {project_path}[/green]")
    else:
        console.print(f"[red]删除失败或项目不存在: {project_path}[/red]")


@cli.command()
@click.pass_context
def cleanup(ctx):
    """清理存储空间"""
    client = ctx.obj['client']
    
    console.print("[yellow]清理存储空间...[/yellow]")
    client.cleanup_storage()
    console.print("[green]清理完成![/green]")


@cli.command()
@click.option('--key', help='配置键')
@click.option('--value', help='配置值')
@click.option('--list', 'list_config', is_flag=True, help='列出所有配置')
@click.pass_context
def config(ctx, key, value, list_config):
    """管理配置"""
    client = ctx.obj['client']
    
    if list_config:
        # 显示所有配置
        config_data = client.config.config
        
        table = Table(title="配置信息")
        table.add_column("配置键", style="cyan")
        table.add_column("配置值", style="green")
        
        for k, v in config_data.items():
            if k == 'openai_api_key' and v:
                v = '*' * 8 + v[-4:] if len(v) > 4 else '*' * len(v)
            table.add_row(k, str(v))
        
        console.print(table)
    
    elif key and value:
        # 设置配置
        client.config.set(key, value)
        console.print(f"[green]已设置 {key} = {value}[/green]")
    
    elif key:
        # 获取配置
        value = client.config.get(key)
        if key == 'openai_api_key' and value:
            value = '*' * 8 + value[-4:] if len(value) > 4 else '*' * len(value)
        console.print(f"{key}: {value}")
    
    else:
        console.print("[yellow]请指定 --key 和 --value 来设置配置，或使用 --list 查看所有配置[/yellow]")


if __name__ == '__main__':
    import sys
    sys.argv = ['main.py', 'index', 'example_project']
    cli()