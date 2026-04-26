import cProfile
import pstats
from random import choice
import json
import httpx
import asyncio
import aiofiles
import time 
from parsel import Selector
from bs4 import BeautifulSoup
from tenacity import retry, wait_exponential, stop_never

class PergamoSpider:
    def __init__(self, user_agents, delay=1.0):
       
        self.base_url = "https://pergamos.lib.uoa.gr"
        
        self.start_url = "https://pergamos.lib.uoa.gr/uoa/dl/frontend/el/browse/1058836?p.tpl=list"
        
        self.urls_output_file = 'pergamos_pdf_urls.json'  # JSON for URLs
        
        self.text_output_file = 'pergamos_text_output.txt'  # Plain text for scraped content
        
        self.user_agents = user_agents
        
        self.delay = delay
        
        self.client = None
        
        self.urls_to_write = []

    # Use Tenacity retry decorator with indefinite retry, exponential backoff, and wait before retry
    @retry(stop=stop_never, wait=wait_exponential(min=2, max=10))
    
    async def fetch_and_parse(self, url):
        
        await asyncio.sleep(self.delay)  # Wait for the specified delay before making a request
        
        headers = {'User-Agent': choice(self.user_agents)}  # Choose a random user agent for each request

        try:
            response = await self.client.get(url, headers=headers, timeout=60.0)  # longer timeout
        
            text = response.text
            
            return await self.parse(text)
        
        except (httpx.ReadTimeout, httpx.HTTPStatusError) as e:
            
            print(f"Error fetching {url}: {e}. Retrying after delay...")
            
            raise  # Raise the exception to trigger Tenacity's retry logic

    async def parse(self, text):
        
        selector = Selector(text)
        
        urls = []
        
        for e in selector.css('.masonry-container > div.well'):
        
            href = e.css('p:nth-child(1) > big:nth-child(2) > a:nth-child(1)::attr(href)').get()
            
            if href:
            
                author = e.css('div:nth-child(4) > div:nth-child(2) > div:nth-child(1) > strong:nth-child(1)::text').get()
                
                if author == "Συγγραφέας:":
                    full_url = f"{self.base_url}{href}"
                    print(full_url)
                    self.urls_to_write.append({'url': full_url})
                else:
                    urls.append(f"{self.base_url}{href}")
        arrow = selector.css(".page-header > div:nth-child(1) > div:nth-child(2) > a:nth-child(3)::attr(href)").get()
        
        if arrow:
            urls.append(f"{self.base_url}{arrow}")
        return urls

    # Use Tenacity's retry decorator for retries when fetching text content
    @retry(stop=stop_never, wait=wait_exponential(min=2, max=10))
    
    async def textractor(self):
    
        async with aiofiles.open(self.text_output_file, mode='a') as file:
        
            for url in self.urls_to_write:
                try:
                    response = await self.client.get(url['url'], timeout=60.0)  # Set a longer timeout
                    if response.status_code != 200:
                        print(f"Failed to fetch page. Status code: {response.status_code}")
                        continue  # Skip to the next URL

                    soup = BeautifulSoup(response.text, "html.parser")
                    elements = soup.select("body > div.content-block-sm > div > div")
                    all_text = [element.get_text(strip=True) for element in elements]
                    cleaned_text = [text for text in all_text if text]

                    if cleaned_text:
                        await file.write(f"URL: {url['url']}\n")
                        await file.write("\n".join(cleaned_text) + "\n\n")
                        print(f"Saved text from: {url['url']}")
                except Exception as e:
                    print(f"Error fetching {url['url']}: {e}")
                    continue  # Skip to the next URL

        print(f"Text from all pages has been saved to {self.text_output_file}")

    async def start_requests(self):
        
        async with httpx.AsyncClient() as client:
            self.client = client
            urls = [self.start_url]
            while urls:
                tasks = [asyncio.create_task(self.fetch_and_parse(url)) for url in urls]
                urls = []
                results = await asyncio.gather(*tasks)
                for new_urls in results:
                    for url in new_urls:
                        if url not in urls:
                            urls.append(url) 
            await self.textractor()

async def load_user_agents():
    async with aiofiles.open('user_agents.json', mode='r') as file:
        content = await file.read()
        data = json.loads(content)
        return data['user_agents']

async def main():
    user_agents = await load_user_agents()
    spider = PergamoSpider(user_agents, delay=0.1)
    await spider.start_requests()

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    asyncio.run(main())

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats('pergamos10.prof')
