import requests
from pathlib import Path
import re
from bs4 import BeautifulSoup
import json
from typing import Optional, Dict, List


class DocumentDownloader:
    def __init__(self, output_dir: str = "downloaded_docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Headers to mimic a browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def download_wiki_legal_articles(self) -> Optional[Dict[str, str]]:
        """
        Downloads legal articles from Wikipedia using their API
        Returns: Dictionary of article titles and their text content
        """
        # List of legal articles to download
        articles = {
            "common_law": "Common_law",
            "constitutional_law": "Constitutional_law",
            "criminal_law": "Criminal_law",
            "civil_law": "Civil_law_(common_law)",
            "contract_law": "Contract",
            "property_law": "Property_law",
            "tort_law": "Tort",
            "international_law": "International_law",
        }

        base_url = "https://en.wikipedia.org/w/api.php"

        try:
            content_dict = {}
            for name, article in articles.items():
                # Parameters for the Wikipedia API
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": article,
                    "prop": "extracts",
                    "explaintext": True,  # Get plain text instead of HTML
                    "exsectionformat": "plain",
                }

                response = requests.get(base_url, params=params, headers=self.headers)
                response.raise_for_status()

                data = response.json()
                # Navigate the response structure
                pages = data["query"]["pages"]
                page_id = list(pages.keys())[0]  # Get the first (and only) page ID

                if "extract" in pages[page_id]:
                    text_content = pages[page_id]["extract"]
                    content_dict[name] = text_content

                    # Save individual article
                    output_file = self.output_dir / f"wiki_{name}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(text_content)

            # Save complete collection
            if content_dict:
                complete_output = self.output_dir / "wiki_legal_complete.txt"
                with open(complete_output, "w", encoding="utf-8") as f:
                    for name, content in content_dict.items():
                        f.write(f"\n\n{'='*20} {name.upper()} {'='*20}\n\n")
                        f.write(content)

                print(f"Successfully downloaded Wikipedia legal articles to {self.output_dir}")
                return content_dict
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error downloading Wikipedia articles: {e}")
            return None

    def download_legal_documents(self) -> Optional[Dict[str, str]]:
        """
        Downloads structured legal information from Wikidata using their API
        Returns: Dictionary of document titles and their text content
        """
        base_url = "https://www.wikidata.org/w/api.php"

        # List of Wikidata Q-IDs for legal concepts
        legal_concepts = {
            "due_process": "Q185126",
            "habeas_corpus": "Q180475",
            "judicial_review": "Q1425010",
            "precedent": "Q1137033",
            "rule_of_law": "Q160795",
        }

        try:
            content_dict = {}
            for name, qid in legal_concepts.items():
                # Parameters for the Wikidata API
                params = {
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": qid,
                    "languages": "en",
                    "props": "labels|descriptions|claims",
                }

                response = requests.get(base_url, params=params, headers=self.headers)
                response.raise_for_status()

                data = response.json()
                if "entities" in data and qid in data["entities"]:
                    entity = data["entities"][qid]

                    # Extract relevant information
                    label = entity.get("labels", {}).get("en", {}).get("value", "")
                    description = entity.get("descriptions", {}).get("en", {}).get("value", "")
                    claims = entity.get("claims", {})

                    # Format the content
                    content = f"Title: {label}\n\nDescription: {description}\n\nDetails:\n"

                    # Add important statements
                    if "P31" in claims:  # instance of
                        content += "\nInstance of:\n"
                        for claim in claims["P31"]:
                            if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                                content += f"- {claim['mainsnak']['datavalue'].get('value', {}).get('id', '')}\n"

                    content_dict[name] = content

                    # Save individual document
                    output_file = self.output_dir / f"legal_{name}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(content)

            # Save complete collection
            if content_dict:
                complete_output = self.output_dir / "legal_concepts_complete.txt"
                with open(complete_output, "w", encoding="utf-8") as f:
                    for name, content in content_dict.items():
                        f.write(f"\n\n{'='*20} {name.upper()} {'='*20}\n\n")
                        f.write(content)

                print(f"Successfully downloaded legal concepts to {self.output_dir}")
                return content_dict
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error downloading legal concepts: {e}")
            return None

    def download_python_docs(self, version: str = "3.12") -> Optional[Dict[str, str]]:
        """
        Downloads Python documentation and converts to text
        Returns: Dictionary of section names and their text content
        """
        sections = {
            "introduction": f"https://docs.python.org/{version}/tutorial/appetite.html",
            "control_flow": f"https://docs.python.org/{version}/tutorial/controlflow.html",
            "data_structures": f"https://docs.python.org/{version}/tutorial/datastructures.html",
            "modules": f"https://docs.python.org/{version}/tutorial/modules.html",
            "classes": f"https://docs.python.org/{version}/tutorial/classes.html",
            "errors": f"https://docs.python.org/{version}/tutorial/errors.html",
        }

        try:
            content_dict = {}
            for section_name, url in sections.items():
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                # Find main content div
                content = soup.find("div", {"role": "main"})
                if content:
                    # Clean up the text
                    text_content = "\n".join(
                        line.strip() for line in content.get_text().split("\n") if line.strip()
                    )
                    content_dict[section_name] = text_content

                    # Save individual section
                    output_file = self.output_dir / f"python_docs_{section_name}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(text_content)

            # Save complete documentation
            if content_dict:
                complete_output = self.output_dir / "python_docs_complete.txt"
                with open(complete_output, "w", encoding="utf-8") as f:
                    for section, content in content_dict.items():
                        f.write(f"\n\n{'='*20} {section.upper()} {'='*20}\n\n")
                        f.write(content)

                print(f"Successfully downloaded Python docs to {self.output_dir}")
                return content_dict
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error downloading Python documentation: {e}")
            return None

    def download_rfc_standards(self) -> Optional[Dict[str, str]]:
        """
        Downloads select IETF RFC documents (more accessible than IEEE)
        Returns: Dictionary of RFC documents and their content
        """
        # Selected important RFCs
        rfcs = {
            "http": "https://www.rfc-editor.org/rfc/rfc2616.txt",  # HTTP/1.1
            "json": "https://www.rfc-editor.org/rfc/rfc8259.txt",  # JSON
            "oauth": "https://www.rfc-editor.org/rfc/rfc6749.txt",  # OAuth 2.0
        }

        try:
            content_dict = {}
            for name, url in rfcs.items():
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()

                # RFCs are already in plain text format
                text_content = response.text
                content_dict[name] = text_content

                # Save individual RFC
                output_file = self.output_dir / f"rfc_{name}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text_content)

            # Save complete RFC collection
            if content_dict:
                complete_output = self.output_dir / "rfc_standards_complete.txt"
                with open(complete_output, "w", encoding="utf-8") as f:
                    for name, content in content_dict.items():
                        f.write(f"\n\n{'='*20} RFC - {name.upper()} {'='*20}\n\n")
                        f.write(content)

                print(f"Successfully downloaded RFC Standards to {self.output_dir}")
                return content_dict
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error downloading RFC Standards: {e}")
            return None

    def download_cc_licenses(self) -> Optional[Dict[str, str]]:
        """
        Downloads Creative Commons license texts
        Returns: Dictionary of license texts
        """
        licenses = {
            "by": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
            "by-sa": "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt",
            "by-nc": "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt",
        }

        try:
            content_dict = {}
            for name, url in licenses.items():
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()

                text_content = response.text
                content_dict[name] = text_content

                # Save individual license
                output_file = self.output_dir / f"cc_license_{name}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text_content)

            # Save complete license collection
            if content_dict:
                complete_output = self.output_dir / "cc_licenses_complete.txt"
                with open(complete_output, "w", encoding="utf-8") as f:
                    for name, content in content_dict.items():
                        f.write(f"\n\n{'='*20} CC LICENSE - {name.upper()} {'='*20}\n\n")
                        f.write(content)

                print(f"Successfully downloaded CC Licenses to {self.output_dir}")
                return content_dict
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error downloading CC Licenses: {e}")
            return None


def download_all_documents(output_dir: str = "downloaded_docs") -> Dict[str, str]:
    """
    Downloads all documents and returns their text content
    Returns: Dictionary with text content of all documents
    """
    downloader = DocumentDownloader(output_dir)

    results = {
        "wiki_legal": downloader.download_wiki_legal_articles(),
        # 'legal_concepts': downloader.download_legal_documents(),
        "python_docs": downloader.download_python_docs(),
        "rfc_standards": downloader.download_rfc_standards(),
        "cc_licenses": downloader.download_cc_licenses(),
    }

    return {k: v for k, v in results.items() if v is not None}


if __name__ == "__main__":
    # Or download all at once
    all_docs = download_all_documents()

    print("\nDownloaded documents:")
    for doc_name, content in all_docs.items():
        if isinstance(content, dict):
            print(f"{doc_name}: {len(content)} sections")
        else:
            print(f"{doc_name}: {len(content)} characters")
