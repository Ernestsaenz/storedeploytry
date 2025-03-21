# components/rag_chain.py
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic  # Only new import needed

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
from pathlib import Path
import os  # Added for environment variable access

class RAGChain:
    def __init__(self):
        self.persist_directory = "./chroma_db"
        self.collection_name = "hepatology_docs"
        # Only change is here - replacing ChatOpenAI with ChatAnthropic
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-r1:free",  
            #perplexity/r1-1776
            #google/gemini-2.0-flash-thinking-exp:free
            #
            
            # The DeepSeek R1 1776 model
            temperature=0.2,
            max_tokens=6000,  # Note: changed from max_tokens_to_sample
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://replit.com",
                ## Replace with your real site
                "X-Title": "Coeliac Disease Expert System"
            }
        )
        # self.llm = ChatAnthropic(
          #  model="claude-3-5-sonnet-20241022",
           # temperature=0.2,
            #max_tokens_to_sample=6000,  # Changed from max_tokens to max_tokens_to_sample for Claude          anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        
        self.embedding_function = OpenAIEmbeddings()  # Keep OpenAI embeddings
        self.db = None
        self.qa_chain = None

    def collection_exists(self):
        """Check if collection already exists"""
        if not Path(self.persist_directory).exists():
            return False
        try:
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
            collection = db.get()
            return collection['ids'] != []
        except Exception:
            return False

    def initialize(self, document_generator):
        """Initialize or load vector store with persistence"""
        print("Checking for existing vector store...")

        if self.collection_exists():
            print("Loading existing vector store...")
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
        else:
            print("Creating new vector store...")
            # Process first batch
            first_batch = next(document_generator)
            self.db = Chroma.from_documents(
                first_batch[:100],
                self.embedding_function,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            print("Processed initial 100 documents")
            time.sleep(30)

            # Process remaining documents
            if len(first_batch) > 100:
                self.db.add_documents(first_batch[100:])
                print("Completed first batch")
                time.sleep(30)

            # Process remaining batches
            for i, batch in enumerate(document_generator, 1):
                print(f"Processing batch {i+1}...")
                for j in range(0, len(batch), 100):
                    chunk = batch[j:j+100]
                    try:
                        self.db.add_documents(chunk)
                        print(f"Processed {len(chunk)} documents")
                        time.sleep(70)
                    except Exception as e:
                        if "rate_limit" in str(e).lower():
                            print("Rate limit hit, waiting 70 seconds...")
                            time.sleep(70)
                            self.db.add_documents(chunk)
                print(f"Completed batch {i+1}")
                # Persist after each batch
                self.db.persist()
                time.sleep(70)

        # Set up retriever and chain
#        retriever = self.db.as_retriever(
#            search_type="similarity", 
#            search_kwargs={"k": 10}
#        )

        retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,        # Number of documents to finally return
                "fetch_k": 20,  # Number of initial candidates to fetch
                "lambda_mult": 0.5  # 0.5 balances relevance and diversity
            }
        )
        # Your existing template
        template = """
        Forget all previous instructions.

        Role: World-Class Expert Clinical Consultant in Coeliac Disease and Gluten-Related Disorders.
        You are a highly capable, thoughtful, and precise globally recognized expert gastroenterologist with extensive experience in diagnosing, managing, and treating Coeliac Disease (CD) and related gluten disorders. Think step-by-step through complex clinical questions and problems. Your responses must be evidence-based and strictly grounded in the clinical guidelines and safety documents provided below. If the available evidence is insufficient for a definitive answer, explicitly state that further evaluation is needed.

        Audience:
        - **Primary:** Gastroenterologists, general practitioners, and clinical specialists involved in the diagnosis and management of gastrointestinal and immune-mediated disorders.
        - **Secondary:** Advanced practice providers, dietitians, nutritionists, and healthcare professionals seeking detailed clinical insights on Coeliac Disease and gluten-related disorders.

        Focus Areas:
        - **Diseases:** Coeliac Disease (CD), Non-Coeliac Gluten Sensitivity (NCGS), Dermatitis Herpetiformis (DH), Refractory Coeliac Disease (RCD)
        - **Clinical Content:**
          - Diagnostic processes (e.g., serological markers, genetic testing, duodenal histopathology with Marsh Classification)
          - Differential diagnoses (e.g., distinguishing CD from NCGS, wheat allergy, IBS, and SIBO)
          - Evidence-based management strategies and treatment protocols (e.g., strict lifelong gluten-free diet, management of refractory CD)
          - Patient education, multidisciplinary care, and monitoring (including nutritional assessments and follow-up care)
          - Recent advancements (e.g., non-invasive adherence tests, emerging therapies, and updated guideline implications)
          - **Medication Safety:** Verify that any medication recommendations are consistent with the FDA’s safety guidelines provided in the document labeled MED-FDA (medication_warnings_before_administration_total.txt).

        Workflow must be:
        1. **Step-by-Step Reasoning:**
           - **Step 1:** Carefully analyze the user's question to identify its clinical aspects.
           - **Step 2:** Determine which parts of the provided context and guidelines (ACG, BSG, ESsCD, and MED-FDA) are relevant.
           - **Step 3:** Develop a structured outline for your response using clear headings, bullet points, and numbered lists.
           - **Step 4:** Draft your evidence-based answer.
        2. **Medication Safety Verification:**
           - If your answer includes a recommendation for a medication, cross-check the medication details with the FDA safety recommendations in MED-FDA.
           - If a specific medication is mentioned, provide any relevant safety recommendations.
           - If no medication is mentioned, do not include any medication safety information.
        3. **Final Answer:**
           - Ensure that the final response is comprehensive, clinically accurate, and solely based on the provided context and evidence.
           - If any uncertainty exists or if additional evidence is needed, clearly indicate so.

        Instructions:
        - Base your response solely on the information in the "Context" section and from the following evidence-based sources:
          - *American College of Gastroenterology Guidelines (ACG): Diagnosis and Management of Celiac Disease*
          - *British Society of Gastroenterology Guidelines (BSG): Diagnosis and Management of Adult Coeliac Disease*
          - *European Society for the Study of Coeliac Disease (ESsCD): Guideline for Coeliac Disease and Gluten-Related Disorders*
          - *FDA Medication Safety Recommendations (MED-FDA)*
        - Structure your answer clearly with headings and lists.
        - For medication recommendations, explicitly verify that they align with the MED-FDA safety information.
        - Always prioritize being truthful, do not generate or assume information that is not supported by the provided context.
        - If further evidence or clarification is needed, indicate which guideline supports your recommendation.
        - Prioritize patient safety, clinical accuracy, and evidence-based practice.
        - When a specific guideline (ACG, BSG, ESsCD, or MED-FDA) is mentioned in the question, provide information from that guideline if available, while also including relevant details from the other guidelines to offer a comprehensive answer. Always cite guidelines by their abbreviations.

        Context:
        {context}

        Question:
        {question}
        """


        
        # Your existing template
        prompt = ChatPromptTemplate.from_template(template)

        self.qa_chain = (
            {
                "context": retriever | self.format_documents,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        print("RAG chain setup complete!")

#    @staticmethod
#    def format_documents(documents):
#        """Format retrieved documents with clear separation"""
#        formatted_docs = []
#        for i, doc in enumerate(documents, 1):
#            formatted_docs.append(f"Document {i}:\n{doc.page_content}\n")
#        return "\n".join(formatted_docs)

    @staticmethod
    def format_documents(documents):
        """Format retrieved documents with clear separation and source info"""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            # Get the source if available, otherwise use numeric identifier
            source = doc.metadata.get("source", f"Document {i}")
            # Get full source name for better readability
            if source == "ACG":
                source_name = "American College of Gastroenterology Guidelines"
            elif source == "BSG":
                source_name = "British Society of Gastroenterology Guidelines"
            elif source == "ESsCD":
                source_name = "European Society for Coeliac Disease Guidelines"
            elif source == "MED-FDA":
                source_name = "Medication Warnings from FDA"
            else:
                source_name = f"Document {i}"

            formatted_docs.append(f"{source_name} (Reference: {source}):\n{doc.page_content}\n")
        return "\n".join(formatted_docs)
    
    def query(self, question: str) -> str:
        """Query the RAG system with error handling and retries"""
        if not self.qa_chain:
            raise ValueError("RAG chain not initialized! Please call initialize() first.")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Processing query with LLM (attempt {attempt + 1})")
                response = self.qa_chain.invoke(question)
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30
                    print(f"Error during query (attempt {attempt + 1}): {str(e)}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    error_msg = str(e)
                    print(f"Final error during query: {error_msg}")
                    return f"An error occurred while processing your question: {error_msg}"
