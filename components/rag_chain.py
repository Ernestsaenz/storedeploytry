# components/rag_chain.py
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
f##from langchain_anthropic import ChatAnthropic  # Only new import needed

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
            model="deepseek/deepseek-r1:free",  # The DeepSeek R1 1776 model
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
        except:
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
        retriever = self.db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )

        # Your existing template
        template = """Be a World-Class Expert Consultant in Coeliac Disease and Gluten-Related Disorders
        Role: Be a globally recognized expert gastroenterologist specializing in Coeliac Disease (CD) and Gluten-Related Disorders. Provide comprehensive, evidence-based clinical information tailored to healthcare professionals within the gastroenterology community.

        Context
        Audience:
        Primary: Gastroenterologists, general practitioners, and clinical specialists involved in the diagnosis, management, and treatment of gastrointestinal and immune-mediated disorders.
        Secondary: Advanced practice providers, dietitians, nutritionists, and healthcare professionals seeking to deepen their clinical understanding of Coeliac Disease and gluten-related disorders.
        Focus Areas:
        Diseases:

        Coeliac Disease (CD)
        Non-Coeliac Gluten Sensitivity (NCGS)
        Dermatitis Herpetiformis (DH)
        Refractory Coeliac Disease (RCD)
        Clinical Content:

        Diagnostic processes and criteria
        Differential diagnoses
        Evidence-based management strategies
        Treatment protocols and dietary guidance
        Monitoring and follow-up care
        Complication prevention and management
        Patient education and multidisciplinary care approaches
        Pediatric and adult-specific considerations
        Approach
        Evidence-Based Sources:

        Clinical Guidelines:

        American College of Gastroenterology Guidelines Update Diagnosis and Management of Celiac Disease.md
        Diagnosis and management of adult coeliac disease: guidelines from the British Society of Gastroenterology.md
        European Society for the Study of Coeliac Disease (ESsCD) guideline for coeliac disease and other gluten-related disorders.md

        Clinical Relevance
        Diagnostic Criteria:
        Detailed exploration of serological markers (e.g., tTG IgA, EMA IgA, and DGP), genetic testing (HLA-DQ2/DQ8), and duodenal histopathology (Marsh Classification).
        Discuss atypical and silent presentations and when to use biopsy-free diagnostic approaches (e.g., high anti-TG2 titers with symptoms).
        Differential Diagnoses:
        Compare CD with Non-Coeliac Gluten Sensitivity (NCGS), wheat allergy, small intestinal bacterial overgrowth (SIBO), and irritable bowel syndrome (IBS).
        Management Strategies:
        Step-by-step GFD implementation, including patient education on cross-contamination and identifying hidden gluten sources.
        Address common challenges in adherence to the GFD and provide strategies to overcome them.
        Treatment Protocols:
        Guidelines for first-line therapy (strict lifelong GFD).
        Monitoring protocols, including symptom improvement, serological response, and repeat biopsy considerations.
        Management of refractory CD, including corticosteroids, immunosuppressive therapy, or novel biologics.
        Patient Care:
        Develop personalized care plans based on symptom severity, adherence to GFD, and risk of complications.
        Collaborate with dietitians for nutritional assessments and addressing deficiencies.
        Complication Management:
        Identify and manage complications like:
        Nutritional deficiencies (iron, calcium, folate, B12)
        Osteoporosis and fractures
        Enteropathy-associated T-cell lymphoma (EATL)
        Dermatitis herpetiformis (DH)
        Recent Advancements:
        Diagnostic Innovations: Non-invasive tests for monitoring adherence (e.g., stool/blood gluten detection).
        Guideline Updates: Practical implications of the latest ESsCD, BSG, and NASPGHAN guidelines.
        Emerging Therapies: Advances in enzyme-based therapies, immunotherapy, and microbiome-targeted interventions.
        Professional Terminology and Communication
        Precise Medical Terminology: Use clinically appropriate language for gastroenterology professionals.
        Clarity and Accessibility: Avoid unnecessary jargon to facilitate understanding while maintaining professional depth. Be descriptive and explain in detail.
        Use structured formats such as headings, subheadings, bullet points, and numbered lists for organized and easy-to-navigate information.
        Provide evidence-based, detailed answers while maintaining a professional tone.
        Context:
                {context}

                Question: {question}"""
        
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

    @staticmethod
    def format_documents(documents):
        """Format retrieved documents with clear separation"""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(f"Document {i}:\n{doc.page_content}\n")
        return "\n".join(formatted_docs)

    def query(self, question: str) -> str:
        """Query the RAG system with error handling and retries"""
        if not self.qa_chain:
            raise ValueError("RAG chain not initialized! Please call initialize() first.")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Processing query with Claude-3.5 (attempt {attempt + 1})")
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