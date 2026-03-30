import re

from .base import FilterTextOperation

class ResponseCleaningFilter(FilterTextOperation):
    def __init__(self):
        super().__init__("filter_clean")
        self.pattern = None
        
    async def start(self):
        await super().start()
        self.pattern = re.compile(r"\[[^\[\]]+\]:\s*")
        
    async def close(self):
        await super().close()
    
    async def configure(self, config_d):
        '''Configure and validate operation-specific configuration'''
        return
    
    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {}

    async def _generate(self, content: str = None, **kwargs):
        '''Generate a output stream'''
        if not content:
            return

        # 1. Remover etiquetas de nomes [NOME]:
        content = self.pattern.sub("", content)
        
        # 2. Remover pensamentos/ações entre parênteses: (pensando...)
        content = re.sub(r"\(.*?\)", "", content)
        
        # 3. Remover tags de comandos especiais: [[emote:xyz]]
        content = re.sub(r"\[\[.*?\]\]", "", content)
        
        # 4. Remover Emojis (blocos de caracteres especiais do Unicode)
        # Fonte: Conjunto abrangente de emojis
        content = re.sub(
            u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251\U0001f926-\U0001f937\U00010000-\U0010ffff"
            u"\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+",
            '', content, flags=re.UNICODE
        )

        # 5. Fatiar o texto em frases para criar cadência (pausas naturais entre falas)
        # O Edge TTS e o RVC processarão frase por frase, gerando pausas de "respiração".
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                yield {
                    "content": sentence
                }

