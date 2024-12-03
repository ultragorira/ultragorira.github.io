# Agentic Scraping for your Lego Set

![BaradDur](/images/Agentic_Scraping/BaradDur.jpg)


**Disclaimer: Your partner may not like this blog post.**

In this blog post I will show you how you can convince your partner that the Lego set you are craving for is worth buying. Which isn't btw?
We will explore agentic scraping with [AgentQL](https://www.agentql.com/) and then analyze the scraped data with a BERT tuned on tweets to get the sentiment. If the majority of the data is positive, we will buy the set. 


### Scraping the data

If you have done some scraping, you know very well that it can be pretty frustrating and annoying to do as every website is different and there is no just one way to scrape any website. However, what if you could leverage Agents and LLM to do the scraping in an easier way. One tool I found for this exercise is AgentQL. The usage is pretty straightforward, and it has a good free-tier for you to try it out. The only thing you need is to create an API key for you to start playing. 

You can create the API key from [here](https://dev.agentql.com/sign-in?redirect_url=https%3A%2F%2Fdev.agentql.com%2F).

## Libraries used

Below is the list of libraries we will use, along with the dataclasses we will interact with. The most important one is the GraphQLQueries where you basically list the GraphQL queries used by the scraper. For example the COOKIES_FORM will let the Agent click on the cookies form that appears when opening YouTube with playwright. 

```
from typing import List, Dict, Optional, Any, Union
import logging
import agentql
from dataclasses import dataclass
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from playwright.sync_api import Page, Browser, sync_playwright

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@dataclass
class YouTubeComment:
    """Represents a YouTube comment with channel name and text."""
    channel_name: str
    comment_text: str

@dataclass
class YouTubeVideo:
    """Represents a YouTube video with basic metadata."""
    video_link: Any  # Playwright element
    video_title: Any  # Playwright element
    channel_name: str

@dataclass
class SentimentResult:
    """
    Container for sentiment analysis results.
    
    Attributes:
        text: The analyzed text
        sentiment: The predicted sentiment (POSITIVE, NEGATIVE, NEUTRAL)
        score: Confidence score for the prediction
        compound_score: Normalized score between -1 and 1
    """
    text: str
    sentiment: str
    score: float
    compound_score: float

class GraphQLQueries:
    """Container class for all GraphQL queries used in the scraper."""
    
    COOKIES_FORM = """
        {
            cookies_form {
                reject_btn
            }
        }
    """
    
    SEARCH = """
        {
            search_input
            search_btn
        }
    """
    
    VIDEO = """
        {
            videos[] {
                video_link
                video_title
                channel_name
            }
        }
    """
    
    VIDEO_CONTROLS = """
        {
            play_or_pause_btn
            expand_description_btn
        }
    """
    
    DESCRIPTION = """
        {
            description_text
        }
    """
    
    COMMENTS = """
        {
            comments[] {
                channel_name
                comment_text
            }
        }
    """
```

## Create the Scraper

The objective is to obtain feedback on a specific Lego set, Barad Dur. We want to understand what other people think about it and what better place to do this than YouTube. On the official page of AgentQL there is already an example code, the below is just put nicely as a class. 


```
class YouTubeScraper:
    """
    A class to scrape YouTube videos and perform sentiment analysis on comments.
    
    Dependencies:
        - agentql
        - playwright
        - transformers
        - torch
        
    Attributes:
        url (str): The base YouTube URL
        page (Page): The Playwright page object
        sentiment_analyzer: The sentiment analysis model
        scroll_count (int): Number of times to scroll down to load comments
        type_delay (int): Delay between keystrokes when typing in search
    """
    
    def __init__(
        self, 
        url: str = "https://www.youtube.com",
        scroll_count: int = 15,
        type_delay: int = 75
    ):
        """
        Initialize the YouTube scraper.
        
        Args:
            url (str): The base YouTube URL to start from
            scroll_count (int): Number of times to scroll down to load comments
            type_delay (int): Delay between keystrokes when typing in search
        """
        self.url = url
        self.scroll_count = scroll_count
        self.type_delay = type_delay
        self.page: Optional[Page] = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self._playwright = None
        self._browser = None
    
    def setup_browser(self) -> None:
        """Set up the browser and navigate to the initial page."""
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=False)
            self.page = agentql.wrap(self._browser.new_page())
            self.page.goto(self.url)
        except Exception as e:
            log.error(f"Failed to setup browser: {e}")
            self.cleanup()
            raise RuntimeError(f"Browser setup failed: {e}")
    
    def cleanup(self) -> None:
        """Clean up browser resources."""
        if self._browser:
            try:
                self._browser.close()
            except Exception as e:
                log.error(f"Error closing browser: {e}")
        
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception as e:
                log.error(f"Error stopping playwright: {e}")
    
    def handle_cookies(self) -> None:
        """Handle the cookies consent form if present."""
        response = self.page.query_elements(GraphQLQueries.COOKIES_FORM)
        if response.cookies_form.reject_btn is not None:
            response.cookies_form.reject_btn.click()
    
    def search_video(self, search_term: str) -> None:
        """Search for a video using the given search term."""
        response = self.page.query_elements(GraphQLQueries.SEARCH)
        response.search_input.type(search_term, delay=self.type_delay)
        response.search_btn.click()
    
    def click_first_video(self) -> str:
        """Click the first video in search results."""
        response = self.page.query_elements(GraphQLQueries.VIDEO)
        video_title = response.videos[0].video_title.text_content()
        log.debug(f"Clicking Youtube Video: {video_title}")
        response.videos[0].video_link.click()
        return video_title
    
    def expand_description(self) -> str:
        """Expand and get the video description."""
        response = self.page.query_elements(GraphQLQueries.VIDEO_CONTROLS)
        response.expand_description_btn.click()
        
        response_data = self.page.query_data(GraphQLQueries.DESCRIPTION)
        description = response_data['description_text']
        log.debug(f"Captured the following description: \n{description}")
        return description
    
    def scroll_to_comments(self) -> None:
        """Scroll down to load comments."""
        log.debug(f"Scrolling {self.scroll_count} times to load comments...")
        for i in range(self.scroll_count):
            self.page.keyboard.press("PageDown")
            self.page.wait_for_page_ready_state()
            log.debug(f"Completed scroll {i + 1}/{self.scroll_count}")
    
    def get_comments(self) -> List[Dict[str, str]]:
        """Get all loaded comments."""
        response = self.page.query_data(GraphQLQueries.COMMENTS)
        comments = response.get("comments", [])
        log.debug(f"Captured {len(comments)} comments!")
        return comments
    
    def analyze_comments(self, comments: List[Dict[str, str]]) -> Dict:
        """
        Analyze sentiment of comments using the enhanced analyzer.
        
        Args:
            comments: List of comment dictionaries with 'comment_text' field
            
        Returns:
            Dict containing detailed sentiment analysis results
        """
        texts = [comment["comment_text"] for comment in comments]
        results = self.sentiment_analyzer.analyze_texts(texts)
        return self.sentiment_analyzer.get_sentiment_summary(results)
    
    def run(self, search_term: str) -> Dict:
        """
        Run the complete scraping and analysis process.
        
        Args:
            search_term (str): The search term to use for finding a video
            
        Returns:
            Dict: Detailed sentiment analysis results
            
        Raises:
            Exception: If any step in the process fails
        """
        try:
            self.setup_browser()
            self.handle_cookies()
            self.search_video(search_term)
            video_title = self.click_first_video()
            description = self.expand_description()
            self.scroll_to_comments()
            comments = self.get_comments()
            
            sentiment_results = self.analyze_comments(comments)
            
            return {
                "video_title": video_title,
                "total_comments": len(comments),
                "sentiment_analysis": sentiment_results
            }
            
        except Exception as e:
            log.error(f"Found Error: {e}")
            raise e
        finally:
            self.cleanup()

```

Couple of notes here.

scroll_count: int = 15 -> This is the number of times the Agent will scroll through the comment section of the video. 
The scraper will click on the first video that is searched in this case from YouTube.

In the init there is "self.sentiment_analyzer = SentimentAnalyzer()" which is the Sentiment Analysis class we will be creating below so that we can analyze the data.

## Sentiment Analysis Class

The sentiment analysis will be done with a finetuned model from [HuggingFace](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) on English tweets.


```
class SentimentAnalyzer:
    """
     Sentiment analysis using RoBERTa model fine-tuned for sentiment analysis.

    """
    
    def __init__(
        self,
        model_name: str = "finiteautomata/bertweet-base-sentiment-analysis",
        batch_size: int = 8,
        device: Optional[str] = None
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: The pretrained model to use
            batch_size: Number of texts to process simultaneously
            device: Device to run model on ('cuda' or 'cpu'). If None, automatically detected.
        """
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            log.info(f"Loaded model: {model_name}")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise
            
        self.id2label = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE"
        }
    
    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches for efficient processing."""
        return [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before analysis."""
        return text.strip()
    
    def _compute_compound_score(self, scores: np.ndarray) -> float:
        """Compute a compound sentiment score between -1 and 1."""
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=0).numpy()
        weights = np.array([-1.0, 0.0, 1.0])
        return float(np.sum(probs * weights))

    @torch.no_grad()
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for a batch of texts."""
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        inputs = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        predictions = outputs.logits.cpu().numpy()
        
        results = []
        for text, prediction in zip(texts, predictions):
            label_id = np.argmax(prediction)
            sentiment = self.id2label[label_id]
            score = float(np.max(torch.nn.functional.softmax(torch.tensor(prediction), dim=0).numpy()))
            compound = self._compute_compound_score(prediction)
            
            results.append(SentimentResult(
                text=text,
                sentiment=sentiment,
                score=score,
                compound_score=compound
            ))
        
        return results
    
    def analyze_texts(self, texts: Union[str, List[str]]) -> Union[SentimentResult, List[SentimentResult]]:
        """Analyze sentiment for one or more texts."""
        single_text = isinstance(texts, str)
        texts_to_process = [texts] if single_text else texts
        
        all_results = []
        for batch in self._batch_texts(texts_to_process):
            results = self.analyze_batch(batch)
            all_results.extend(results)
        
        return all_results[0] if single_text else all_results
    
    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict:
        """Generate summary statistics for a set of sentiment results."""
        total = len(results)
        if total == 0:
            return {
                "total_analyzed": 0,
                "sentiment_distribution": {},
                "dominant_sentiment": "N/A",
                "average_compound_score": 0.0,
                "confidence": {"high_confidence": 0, "medium_confidence": 0, "low_confidence": 0}
            }
            
        sentiment_counts = {
            "POSITIVE": sum(1 for r in results if r.sentiment == "POSITIVE"),
            "NEUTRAL": sum(1 for r in results if r.sentiment == "NEUTRAL"),
            "NEGATIVE": sum(1 for r in results if r.sentiment == "NEGATIVE")
        }
        
        avg_compound = sum(r.compound_score for r in results) / total
        
        return {
            "total_analyzed": total,
            "sentiment_distribution": {
                k: f"{(v/total)*100:.1f}%" for k, v in sentiment_counts.items()
            },
            "dominant_sentiment": max(sentiment_counts.items(), key=lambda x: x[1])[0],
            "average_compound_score": avg_compound,
            "confidence": {
                "high_confidence": sum(1 for r in results if r.score > 0.8),
                "medium_confidence": sum(1 for r in results if 0.5 < r.score <= 0.8),
                "low_confidence": sum(1 for r in results if r.score <= 0.5)
            }
        }

```

Now that we have the two classes created, let's put it into action and see what happens


## LET'S SCRAPE

```
if __name__ == "__main__":

    scraper = YouTubeScraper()
    video_analysis_results = scraper.run("Lego Barad Dur Review")
    
    print(f"\nVideo: {video_analysis_results['video_title']}")
    print(f"Total comments analyzed: {video_analysis_results['sentiment_analysis']['total_analyzed']}")
    print("\nSentiment Distribution:")

    for sentiment, percentage in video_analysis_results['sentiment_analysis']['sentiment_distribution'].items():
        print(f"{sentiment}: {percentage}")
    
    print(f"\nDominant Sentiment: {video_analysis_results['sentiment_analysis']['dominant_sentiment']}")
    print(f"Average Sentiment Score: {video_analysis_results['sentiment_analysis']['average_compound_score']:.2f}")
    
    confidence = video_analysis_results['sentiment_analysis']['confidence']
    print("\nConfidence Levels:")
    print(f"High confidence predictions: {confidence['high_confidence']}")
    print(f"Medium confidence predictions: {confidence['medium_confidence']}")
    print(f"Low confidence predictions: {confidence['low_confidence']}")


    if video_analysis_results['sentiment_analysis']['dominant_sentiment'] == "POSITIVE":
        print("YES, BUY THE SET MATE!")
    else:
        print("NO, JUST DO NOT BUT IT...")
```

Let's see it in action:


https://github.com/user-attachments/assets/eba841c1-8223-4aae-891d-d3e6b33b8806



The results:

```
Video: LEGO Lord of the Rings Barad-dur (Review)
Total comments analyzed: 40

Sentiment Distribution:
POSITIVE: 50.0%
NEUTRAL: 42.5%
NEGATIVE: 7.5%

Dominant Sentiment: POSITIVE
Average Sentiment Score: 0.40

Confidence Levels:
High confidence predictions: 30
Medium confidence predictions: 9
Low confidence predictions: 1

YES, BUY THE SET MATE!

```

Well I guess we need to order this set then. :) 
