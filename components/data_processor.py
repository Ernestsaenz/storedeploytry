# components/data_processor.py
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pathlib import Path
import json
import hashlib

class DataProcessor:
    def __init__(self):
        self.urls = [
            "https://www.dropbox.com/scl/fi/giq74bwv8uhr5urd0a0gk/cg9_combined.txt?rlkey=y7j2iaig54enl7cad9vwnz0ae&dl=1",
            "https://www.dropbox.com/scl/fi/hib1wjgamsxg4gbp9wlym/cg10_combined.txt?rlkey=k1lpbbn0as17bumxd3qx6hqtb&dl=1",
            "https://www.dropbox.com/scl/fi/0al9ards4g4d59tblq8ba/cg11_combined.txt?rlkey=8zb2yfkgyqzklhizren55xyny&dl=1",
            "https://www.dropbox.com/scl/fi/jpi330r90tffefc8yrcud/cg12_combined.txt?rlkey=w2xa902zgmd7qw7acmekl803m&dl=1",
            "https://www.dropbox.com/scl/fi/ctl5f3t0pn21ete33es4g/cg13_combined.txt?rlkey=vv7wzjpxt7ec0poizdyw1r258&dl=1",
            "https://www.dropbox.com/scl/fi/th6p955mdpt5bjxgozgyv/cg1_combined.txt?rlkey=83nvn9er49pcvpmgi1s0oieul&dl=1",
            "https://www.dropbox.com/scl/fi/wp9qt8d6x795jlqqxlqcx/cg2_combined.txt?rlkey=4p8c7qdx3idixb5amecmy3nr6&dl=1",
            "https://www.dropbox.com/scl/fi/bwu5z5asmfd0y1d87m0n7/cg3_combined.txt?rlkey=ym5hqnpxjfv6jpd7bgmjpia8s&dl=1",
            "https://www.dropbox.com/scl/fi/v06k8pcyxpz07cneoh5et/cg4_combined.txt?rlkey=q4lprx5vyi7fp6zn2tpm5sc8e&dl=1",
            "https://www.dropbox.com/scl/fi/9lgu141zwjcm7kz500azb/cg5_combined.txt?rlkey=eajdl05zb7id34pj3xece8zpd&dl=1",
            "https://www.dropbox.com/scl/fi/pemcm8mmmgerlu3y0dzer/cg6_combined.txt?rlkey=bfe6322dn13kaibmag20iqljq&dl=1",
            "https://www.dropbox.com/scl/fi/403a2e6qyrcl5896lxkr6/cg7_combined.txt?rlkey=spteu8yy901plpkgnh2niobo9&dl=1",
            "https://www.dropbox.com/scl/fi/tmzxpvlk40l7dqhoh21g3/cg8_combined.txt?rlkey=zfsjx0uhxrji4gj0aasbhck5g&dl=1",
            "https://www.dropbox.com/scl/fi/830vbki4033uy3tgn4h12/medication_warnings_before_administration_total.txt?rlkey=bgx7ntqat420urqzufaje0av7&st=knom5zif&dl=1"
        ]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.batch_size = 250
        self.cache_dir = Path("./document_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_path(self, url):
        """Generate cache file path for a URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.json"

    def fetch_url(self, url):
        """Fetch content from URL with caching"""
        cache_path = self.get_cache_path(url)

        # Check cache first
        if cache_path.exists():
            print(f"Loading cached content for {url}")
            with open(cache_path, 'r') as f:
                return json.load(f)['content']

        # Fetch if not cached
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.text

            # Cache the content
            with open(cache_path, 'w') as f:
                json.dump({'url': url, 'content': content}, f)

            return content
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def process_documents(self):
        """Process documents with improved chunking"""
        all_documents = []

        for url in self.urls:
            print(f"Processing URL: {url}")
            content = self.fetch_url(url)
            if content:
                chunks = self.text_splitter.split_text(content)
                for chunk in chunks:
                    document = Document(page_content=chunk)
                    all_documents.append(document)
                print(f"Successfully processed {len(chunks)} chunks from {url}")
            else:
                print(f"Failed to process {url}")

        print(f"Total documents collected: {len(all_documents)}")

        # Yield documents in batches
        for i in range(0, len(all_documents), self.batch_size):
            batch = all_documents[i:i + self.batch_size]
            yield batch