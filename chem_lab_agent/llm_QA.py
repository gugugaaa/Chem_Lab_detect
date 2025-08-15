import os
import sys
from typing import List, Dict, Any, Optional, Union, Callable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChemLabAgent:
    """化学实验室智能助手，基于RAG架构，使用Cohere embedding和DeepSeek LLM"""
    
    def __init__(self):
        # 检查并设置API密钥
        self._check_api_keys()
        
        # 初始化模型和向量存储
        self.embeddings = CohereEmbeddings(
            model="embed-multilingual-light-v3.0",
            cohere_api_key=os.environ["COHERE_API_KEY"]
        )
        
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=2000,
            timeout=None,
            max_retries=2,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            streaming=True  # 添加streaming=True以启用流式输出
        )
        
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
    
    def _check_api_keys(self) -> None:
        """检查必要的API密钥是否存在"""
        if not os.environ.get("COHERE_API_KEY"):
            logger.error("未找到COHERE_API_KEY环境变量")
            raise ValueError("请设置COHERE_API_KEY环境变量")
            
        if not os.environ.get("DEEPSEEK_API_KEY"):
            logger.error("未找到DEEPSEEK_API_KEY环境变量")
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
    
    def load_pdf(self, pdf_path: str) -> List[Any]:
        """加载PDF文档并分割成chunks"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            raise FileNotFoundError(f"找不到PDF文件: {pdf_path}")
            
        logger.info(f"加载PDF文件: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"PDF已分割为 {len(chunks)} 个文本块")
        
        return chunks
    
    def create_vector_store(self, documents: List[Any], persist_directory: str = None) -> None:
        """基于文档创建向量存储"""
        logger.info("创建向量存储...")
        
        if persist_directory:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            self.vector_store.persist()
            logger.info(f"向量存储已持久化至: {persist_directory}")
        else:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        logger.info("向量检索器已设置完成")
    
    def load_vector_store(self, persist_directory: str) -> None:
        """从持久化目录加载向量存储"""
        if not os.path.exists(persist_directory):
            logger.error(f"向量存储目录不存在: {persist_directory}")
            raise FileNotFoundError(f"找不到向量存储目录: {persist_directory}")
            
        logger.info(f"从{persist_directory}加载向量存储")
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        logger.info("向量检索器已加载完成")
    
    def setup_qa_chain(self) -> None:
        """设置问答链"""
        if not self.retriever:
            logger.error("检索器尚未初始化，请先创建或加载向量存储")
            raise ValueError("请先创建或加载向量存储")
            
        # 自定义提示模板
        template = """你是一位专业的化学实验室助手，根据给定的上下文回答问题。
        
        上下文信息:
        {context}
        
        问题: {question}
        
        请提供准确、全面且有帮助的回答。如果上下文中没有足够的信息，请清晰说明。回答要简洁但信息丰富。
        """
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 创建QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("问答链设置完成")
    
    def ask(self, query: str) -> Dict:
        """向系统提问"""
        if not self.qa_chain:
            logger.error("问答链尚未设置")
            raise ValueError("请先设置问答链")
            
        logger.info(f"提问: {query}")
        response = self.qa_chain({"query": query})
        
        return {
            "answer": response["result"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        }
    
    def ask_stream(self, query: str) -> Dict:
        """向系统提问并以流式方式返回答案"""
        if not self.qa_chain:
            logger.error("问答链尚未设置")
            raise ValueError("请先设置问答链")
            
        logger.info(f"流式提问: {query}")
        
        # 从检索器获取相关文档
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 构建提示
        template = """你是一位专业的化学实验室助手，根据给定的上下文回答问题。
        
        上下文信息:
        {context}
        
        问题: {question}
        
        请提供准确、全面且有帮助的回答。如果上下文中没有足够的信息，请清晰说明。回答要简洁但信息丰富。"""
        
        messages = [
            ("system", template.format(context=context, question=query)),
            ("human", query)
        ]
        
        # 流式输出
        full_response = ""
        print("回答: ", end="", flush=True)
        
        for chunk in self.llm.stream(messages):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += content
        
        return {
            "answer": full_response,
            "sources": [doc.page_content for doc in docs]
        }


class StreamHandler(BaseCallbackHandler):
    """处理流式输出的回调处理器"""
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """初始化处理器
        
        Args:
            callback: 每次收到新token时调用的函数
        """
        self.callback = callback
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当LLM生成新token时调用"""
        if self.callback:
            self.callback(token)
        else:
            # 如果没有提供回调，直接打印到控制台
            print(token, end="", flush=True)


def main():
    """主函数，示例如何使用ChemLabAgent"""
    agent = ChemLabAgent()
    
    # 示例：加载PDF并创建向量存储
    pdf_path = input("请输入PDF文件路径: ")
    persist_dir = os.path.join(os.path.dirname(pdf_path), "vector_store")
    
    # 判断是否已有向量存储
    if os.path.exists(persist_dir):
        choice = input(f"检测到已有向量存储({persist_dir})，是否重新创建？(y/n): ")
        if choice.lower() == 'y':
            documents = agent.load_pdf(pdf_path)
            agent.create_vector_store(documents, persist_dir)
        else:
            agent.load_vector_store(persist_dir)
    else:
        documents = agent.load_pdf(pdf_path)
        agent.create_vector_store(documents, persist_dir)
    
    # 设置问答链
    agent.setup_qa_chain()
    
    # 交互式问答循环
    print("\n化学实验室智能助手已就绪。输入'exit'退出。")
    while True:
        query = input("\n请输入您的问题: ")
        if query.lower() == 'exit':
            break
            
        try:
            print("\n", end="")
            # 使用流式回答
            result = agent.ask_stream(query)
            
            print("\n\n信息来源:")
            for i, source in enumerate(result["sources"], 1):
                print(f"来源 {i}:")
                print(source[:200] + "..." if len(source) > 200 else source)
                print()
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
