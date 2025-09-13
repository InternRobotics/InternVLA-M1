
# motivation
这个模块是为了将 模型的 推理封装为推理的 websocket 服务。实现模型和测试环境分离
主要实现入口在 real_deployment/deploy/server_policy.py

1. 它使用tools 中的 ./deploy/tools/model_interface.py 来对接模型变化，并处理推理的 逻辑， 比如 history

2. ./tools/websocket_policy_server.py 中将模型变成 server 服务

3. real_deployment/deploy/tools/websocket_policy_client.py 定义在其他地方如果和 server 建立连接，就好像policy 到了本体环境。 
这里是个参考，具体而言模型训练和模型测试的sim 是 解耦关系。 可以参考websocket_policy_client 在 sim 环境中实现 client。

4. real_deployment/deploy/debug_server_policy.py 的出现是用了在server 挂起来是，本地debug 它， 而构建的 本地client。
