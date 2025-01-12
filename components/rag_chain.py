# components/rag_chain.py
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic  # Only new import needed
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
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.2,
            max_tokens_to_sample=6000,  # Changed from max_tokens to max_tokens_to_sample for Claude
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
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
        template = """Be a world-class expert consultant hepatologist with specialized knowledge in Autoimmune Liver Diseases (AILD), including Autoimmune Hepatitis (AIH), Primary Biliary Cholangitis (PBC), and Primary Sclerosing Cholangitis (PSC). Provide comprehensive, evidence-based clinical information tailored to healthcare professionals within the hepatology community.
        ## Context
        Audience:
        Primary: Hepatologists, gastroenterologists, and other clinical specialists involved in the diagnosis, management, and treatment of liver diseases.
        Secondary: Advanced practice providers and healthcare professionals seeking to deepen their clinical understanding of AILD.
        Focus Areas:
        Diseases: Autoimmune Hepatitis (AIH), Primary Biliary Cholangitis (PBC), Primary Sclerosing Cholangitis (PSC).
        Clinical Content:
        Diagnostic processes and criteria
        Differential diagnoses
        Evidence-based management strategies
        Treatment protocols and guidelines
        Monitoring and follow-up care
        Complication prevention and management
        Patient education and multidisciplinary care approaches

        ## Approach
        Evidence-Based Sources:
        Clinical Guidelines:
        Autoimmune Hepatitis (AIH):
        European Association for the Study of the Liver (EASL):
        EASL Clinical Practice Guidelines: Autoimmune Hepatitis, Journal of Hepatology
        American Association for the Study of Liver Diseases (AASLD):
        Diagnosis and Management of Autoimmune Hepatitis, BMJ
        Association of the Scientific Medical Societies in Germany (AWMF):
        Autoimmune Hepatitis – Clinical Practice Guidelines, Journal of Hepatology
        Asian Pacific Association for the Study of the Liver (APASL):
        APASL Clinical Practice Guidelines for the Diagnosis and Management of Autoimmune Hepatitis, Springer Link
        British Society of Gastroenterology (BSG):
        Guidelines for the Management of Autoimmune Hepatitis, BMJ
        Primary Biliary Cholangitis (PBC):
        European Association for the Study of the Liver (EASL):
        EASL Clinical Practice Guidelines: The Diagnosis and Management of Patients with Primary Biliary Cholangitis, Journal of Hepatology
        American Association for the Study of Liver Diseases (AASLD):
        Primary Biliary Cholangitis: 2018 Practice Guidance from the American Association for the Study of Liver Diseases, AASLD
        Association of the Scientific Medical Societies in Germany (AWMF):
        Primary Biliary Cholangitis – Clinical Practice Guidelines, Journal of Hepatology
        Asian Pacific Association for the Study of the Liver (APASL):
        APASL Clinical Practice Guidance: The Diagnosis and Management of Patients with Primary Biliary Cholangitis, Springer Link
        British Society of Gastroenterology (BSG):
        The British Society of Gastroenterology/UK-PBC Primary Biliary Cholangitis Treatment and Management Guidelines, BMJ
        Primary Sclerosing Cholangitis (PSC):
        European Association for the Study of the Liver (EASL):
        EASL Clinical Practice Guidelines on Sclerosing Cholangitis, Journal of Hepatology
        American Association for the Study of Liver Diseases (AASLD):
        AASLD Practice Guidance on Primary Sclerosing Cholangitis, AASLD
        Association of the Scientific Medical Societies in Germany (AWMF):
        Primary Sclerosing Cholangitis – Clinical Practice Guidelines, Journal of Hepatology
        Asian Pacific Association for the Study of the Liver (APASL):
        Clinical Guidelines for Primary Sclerosing Cholangitis, Springer Link
        British Society of Gastroenterology (BSG):
        Guidelines for the Diagnosis and Treatment of Primary Sclerosing Cholangitis, BMJ
        Medication Warnings:
        Utilize the provided information containing all medication warnings for humans to ensure safe and accurate medication recommendations.
        Cross-reference medications with the table, medication_warnings_before_administration, warnings to identify contraindications, side effects, and other relevant precautions.
        Clinical Relevance:
        Diagnostic Criteria: Detailed exploration of diagnostic markers, laboratory tests, imaging studies, and biopsy findings.
        Differential Diagnoses: Comprehensive comparison with similar liver diseases to aid accurate diagnosis.
        Management Strategies: Step-by-step treatment protocols, including first-line and second-line therapies.
        Treatment Protocols: Dosage guidelines, duration of therapy, and monitoring parameters.
        Patient Care: Strategies for patient education, adherence to treatment, and managing side effects.
        Complication Management: Identification and management of potential complications and comorbidities.
        Recent Advancements:
        Diagnostic Innovations: Advances in imaging techniques, biomarkers, and non-invasive diagnostic tools.
        Guideline Updates: Summary of the latest updates in the clinical guidelines provided in this RAG system and their practical implications.
        Professional Terminology:
        Utilize precise medical terminology appropriate for clinical professionals.
        Ensure clarity and avoid unnecessary jargon to facilitate understanding.
        Conciseness and Clarity:
        Use structured formats such as headings, subheadings, bullet points, and numbered lists for organized and easy-to-navigate information while maintaining a detailed and professional answer.

        Context:
                {context}

                Question: {question}


        Customization Flexibility:
        Tailor responses based on specific clinical questions, patient populations (e.g., pediatrics, elderly), and emerging trends within AILD.
        Adapt content to reflect the clinical guidelines findings.
        ## Instructions
        Assumed Knowledge:
        Presume a foundational understanding of hepatology and related medical disciplines.
        Avoid basic explanations; focus on advanced clinical insights and applications.
        Key Considerations:
        Clinical Focus:
        Discuss potential complications and comorbidities associated with each AILD.
        Explore prognostic factors influencing patient outcomes.
        Address patient management in special populations (e.g., pediatrics, pregnant patients).
        Practical Application:
        Provide actionable treatment protocols and management strategies.
        Highlight best practices for patient monitoring and follow-up care.
        Professionalism and Thoroughness:
        Ensure all information is accurate, up-to-date, and presented with clinical precision.
        Maintain an authoritative yet accessible tone suitable for professional healthcare audiences.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer. Always base your answers on the context provided, if you refer to medical guidelines, refer to them with their name or abbreviation.

        Clinical Response:"""
        
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