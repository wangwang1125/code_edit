def get_user_info(user_id):
"""获取用户信息，返回结构化字典格式"""
# 简单的用户信息获取
name = "张三"
age = 25
email = "zhangsan@example.com"
return {'name': name, 'age': age, 'email': email}

class UserManager:
    def __init__(self):
        self.users = []
    
    def add_user(self, name, age):
        user = {"name": name, "age": age}
        self.users.append(user)
        return True
    
    def get_user_count(self):
        return len(self.users)
