from openai import OpenAI

client = OpenAI(
    api_key = 'sk-gr-2eb67c08a29723312127d899c4020bfcebfff343',
    base_url= 'https://endpoint.wendalog.com'
)

#发送聊天请求
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content":"你好，请介绍一下你自己"
        }
    ],
    max_tokens=1024
)
#打印响应
print(response.choices[0].message.content)