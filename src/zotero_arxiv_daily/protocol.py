from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

def _prefers_chinese(language: str) -> bool:
    language = str(language).lower()
    return 'chinese' in language or '中文' in language or language.startswith('zh')


def _contains_chinese(text: str) -> bool:
    return re.search(r'[\u4e00-\u9fff]', text or '') is not None


def _needs_chinese_tldr_retry(tldr: str, language: str) -> bool:
    if not _prefers_chinese(language):
        return False
    if not tldr or not _contains_chinese(tldr):
        return True
    return len(tldr) > 500


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = (
            "Given the following information of a paper, generate a one-sentence TLDR summary.\n"
            f"Primary language/style: {lang}.\n"
            "Output exactly one concise sentence. Do not copy the abstract verbatim.\n"
            "Preserve the paper title, method names, model names, dataset names, metrics, abbreviations, "
            "and established academic terms in English. Do not translate the paper title.\n"
        )
        if _prefers_chinese(lang):
            prompt += (
                "The summary explanation must be mainly Simplified Chinese. "
                "Use English only for names, titles, abbreviations, formulas, datasets, metrics, and established terms.\n\n"
            )
        else:
            prompt += "\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        system_prompt = (
            "You are an assistant who clearly summarizes scientific papers and gives the core idea to the user. "
            f"Write exactly one concise sentence primarily in {lang}. "
            "Do not copy the abstract verbatim. "
            "If the requested language is Chinese, the explanation must be mainly Simplified Chinese, "
            "but keep paper titles, method/model/dataset names, metrics, abbreviations, and established academic terms in English."
        )

        def request_tldr(user_prompt: str) -> str:
            response = openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            return response.choices[0].message.content

        tldr = request_tldr(prompt)
        if _needs_chinese_tldr_retry(tldr, lang):
            logger.warning(f"Generated TLDR is not a Chinese concise summary for {self.url}. Retrying once.")
            retry_prompt = (
                f"{prompt}\n\n"
                "The previous answer was not acceptable because it was not mainly Chinese or was too long. "
                "Rewrite it now as exactly one concise Simplified Chinese sentence. "
                "Keep only key academic terms, method/model/dataset names, metrics, abbreviations, and the paper title in English."
            )
            tldr = request_tldr(retry_prompt)
            if _needs_chinese_tldr_retry(tldr, lang):
                logger.warning(f"Failed to generate an acceptable Chinese TLDR for {self.url}.")
                return "TL;DR 生成失败，请打开 PDF 或查看论文原文摘要。"

        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            if _prefers_chinese(llm_params.get('language', 'English')):
                tldr = "TL;DR 生成失败，请打开 PDF 或查看论文原文摘要。"
            else:
                tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
