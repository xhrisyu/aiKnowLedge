# aiKnowLedge

aiKnowLedge 是一个用于管理知识库和问答聊天机器人的项目，旨在帮助用户高效管理和利用他们的知识库。

## 技术栈

- **[Streamlit](https://streamlit.io/)**: Web 用户界面
- **[FastAPI](https://fastapi.tiangolo.com/)**: 后端接口服务
- **[Qdrant](https://qdrant.tech/)**: 向量数据库，用于存储和检索知识文件
- **[MongoDB](https://www.mongodb.com/)**: 数据库，用于存储知识文件的元数据

## 功能

### 问答聊天机器人

1. **基于知识库回答问题（导入到向量数据库）**:
   - 支持可定制参数 [model name, temperature, stream, top k, similarity threshold, additional context length]。
   - Model Name: LLM 模型名称（例如 gpt-3.5-turbo, gpt-4-turbo, gpt-4o）。
   - Temperature: LLM 温度设置，控制答案的创造性。
   - Stream: 实时输出答案的流。
   - Top k: 从向量数据库检索的前 k 个块。
   - Similarity Threshold: 问答回答检索的相似度下限。
   - Additional Context Length: 检索文档块的扩展词长度。

2. **显示用户问题意图和检索的文档块**:
   - 显示用户问题关键词和意图的分析。
   - 显示 top k 检索的块，包括原文（附加上下文）、page_id、chunk_id 和相似度得分。
   - 提供详细的检索结果，帮助用户理解和验证答案的来源。

### 知识库管理

1. **查看知识库**:
   - 浏览和管理现有知识文件。
   
2. **向知识库添加新文件**:
   - 支持可定制参数 [chunk size, overlap size, separators]。
   - 灵活的文件切块和重叠设置，以实现高效的组织和检索。
   
3. **从知识库删除文件**:
   - 轻松移除不必要的文件，保持知识库整洁有序。
   
4. **将文件添加到向量数据库（用于问答聊天机器人检索）**:
   - 将知识文件索引到向量数据库，以实现高效的问答回答检索。
   
5. **从向量数据库删除文件**:
   - 移除不再需要的索引文件，优化向量数据库的存储和检索性能。

### 知识点和问题生成

上传段落以生成问题和答案。

## 本地运行

1. 创建并激活 Conda 环境：
   ```shell
   conda env create -n aiknowledge python=3.11.4
   conda activate aiknowledge
   ```
   
2. 安装依赖：
   ```shell
   pip install -r requirements.txt
   ```

3. 启动前端 (Streamlit)：
   ```shell
   streamlit run app.py --server.address=127.0.0.1 --server.port=8501
   ```

4. 启动后端 (FastAPI)：
   ```shell
   uvicorn server.api:app --host 127.0.0.1 --port 8500
   ```

## Docker 部署
1. 构建 Docker 镜像：
   ```shell
   docker build -t aiknowledge .
   ```

2. 启动 Docker 容器：
   ```shell
   docker compose up -d
   ```

## 联系我们

如果您有任何问题或建议，请随时与我们联系：[aiknow2023@gmail.com](mailto:aiknow2023@gmail.com)