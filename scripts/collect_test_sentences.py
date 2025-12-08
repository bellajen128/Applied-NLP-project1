"""
從 Reddit 和 Twitter 收集測試句子
目標：100 個真實、多樣化的句子
"""

import praw
import tweepy
import re
import json
from collections import Counter

# ============================================
# Reddit 爬蟲
# ============================================

def collect_from_reddit(client_id, client_secret, user_agent, num_sentences=60):
    """
    從 Reddit 收集句子
    
    需要先註冊 Reddit API:
    https://www.reddit.com/prefs/apps
    """
    print("="*60)
    print("Collecting from Reddit")
    print("="*60)
    
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    
    # 目標 subreddits
    subreddits = [
        'CasualConversation',
        'AskReddit',
        'teenagers',
        'sports',
        'movies',
        'gaming'
    ]
    
    sentences = []
    seen = set()
    
    for subreddit_name in subreddits:
        print(f"\n[Subreddit: r/{subreddit_name}]")
        subreddit = reddit.subreddit(subreddit_name)
        
        # 抓取熱門 posts 的 comments
        for submission in subreddit.hot(limit=10):
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:20]:
                text = comment.body
                
                # 基本過濾
                if not text or len(text) < 20 or len(text) > 150:
                    continue
                
                # 移除 URLs, mentions, etc.
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'www\.\S+', '', text)
                text = re.sub(r'/r/\S+', '', text)
                text = re.sub(r'u/\S+', '', text)
                
                # 切分成句子
                sents = re.split(r'[.!?]+', text)
                
                for sent in sents:
                    sent = sent.strip()
                    
                    # 句子品質檢查
                    if len(sent) < 10 or len(sent) > 100:
                        continue
                    
                    # 避免重複
                    sent_lower = sent.lower()
                    if sent_lower in seen:
                        continue
                    
                    # 過濾問句、不完整句
                    if sent.endswith('?'):
                        continue
                    if sent.count(' ') < 3:  # 至少 4 個詞
                        continue
                    
                    # 檢查是否包含髒話（簡單版）
                    profanity_words = ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch']
                    if any(word in sent_lower for word in profanity_words):
                        continue
                    
                    sentences.append({
                        'sentence': sent,
                        'source': f'reddit_r/{subreddit_name}',
                        'length': len(sent.split())
                    })
                    seen.add(sent_lower)
                    
                    print(f"  ✅ {len(sentences):3}. {sent[:60]}...")
                    
                    if len(sentences) >= num_sentences:
                        break
            
            if len(sentences) >= num_sentences:
                break
        
        if len(sentences) >= num_sentences:
            break
    
    print(f"\n✅ Collected {len(sentences)} sentences from Reddit")
    return sentences


# ============================================
# Twitter 爬蟲 (使用 tweepy v2)
# ============================================

def collect_from_twitter(bearer_token, num_sentences=40):
    """
    從 Twitter 收集句子
    
    需要 Twitter API v2 Bearer Token:
    https://developer.twitter.com/en/portal/dashboard
    """
    print("\n" + "="*60)
    print("Collecting from Twitter")
    print("="*60)
    
    client = tweepy.Client(bearer_token=bearer_token)
    
    # 搜尋關鍵字（日常話題）
    queries = [
        "amazing -is:retweet lang:en",
        "excited -is:retweet lang:en",
        "terrible -is:retweet lang:en",
        "happy -is:retweet lang:en",
        "sad -is:retweet lang:en",
    ]
    
    sentences = []
    seen = set()
    
    for query in queries:
        print(f"\n[Query: {query.split()[0]}]")
        
        try:
            tweets = client.search_recent_tweets(
                query=query,
                max_results=20,
                tweet_fields=['text']
            )
            
            if not tweets.data:
                continue
            
            for tweet in tweets.data:
                text = tweet.text
                
                # 基本過濾
                if len(text) < 20 or len(text) > 150:
                    continue
                
                # 移除 URLs, hashtags, mentions
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'#\S+', '', text)
                text = re.sub(r'@\S+', '', text)
                text = re.sub(r'RT\s+', '', text)
                
                # 切分句子
                sents = re.split(r'[.!]+', text)
                
                for sent in sents:
                    sent = sent.strip()
                    
                    if len(sent) < 10 or len(sent) > 100:
                        continue
                    
                    sent_lower = sent.lower()
                    if sent_lower in seen:
                        continue
                    
                    if sent.endswith('?'):
                        continue
                    if sent.count(' ') < 3:
                        continue
                    
                    # 過濾髒話
                    profanity_words = ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch']
                    if any(word in sent_lower for word in profanity_words):
                        continue
                    
                    sentences.append({
                        'sentence': sent,
                        'source': 'twitter',
                        'length': len(sent.split())
                    })
                    seen.add(sent_lower)
                    
                    print(f"  ✅ {len(sentences):3}. {sent[:60]}...")
                    
                    if len(sentences) >= num_sentences:
                        break
        
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
        
        if len(sentences) >= num_sentences:
            break
    
    print(f"\n✅ Collected {len(sentences)} sentences from Twitter")
    return sentences


# ============================================
# 合併與後處理
# ============================================

def post_process_sentences(sentences):
    """後處理：多樣性檢查、去重"""
    print("\n" + "="*60)
    print("Post-processing")
    print("="*60)
    
    # 統計詞長分佈
    lengths = [s['length'] for s in sentences]
    print(f"\nLength distribution:")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f} words")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    
    # 統計來源
    sources = Counter([s['source'] for s in sentences])
    print(f"\nSource distribution:")
    for source, count in sources.items():
        print(f"  {source}: {count}")
    
    return sentences


# ============================================
# 主函數
# ============================================

def main():
    """
    收集 100 個測試句子
    
    使用方式：
    1. 註冊 Reddit API: https://www.reddit.com/prefs/apps
    2. 註冊 Twitter API: https://developer.twitter.com/
    3. 填入你的 credentials
    """
    
    # ===== 填入你的 API credentials =====
    
    # Reddit API (必填)
    REDDIT_CLIENT_ID = "your_client_id_here"
    REDDIT_CLIENT_SECRET = "your_client_secret_here"
    REDDIT_USER_AGENT = "slangify_test_collector/1.0"
    
    # Twitter API (選填，如果沒有可以只用 Reddit)
    TWITTER_BEARER_TOKEN = "your_bearer_token_here"
    
    # =====================================
    
    all_sentences = []
    
    # 從 Reddit 收集（60 個）
    if REDDIT_CLIENT_ID != "your_client_id_here":
        reddit_sentences = collect_from_reddit(
            REDDIT_CLIENT_ID,
            REDDIT_CLIENT_SECRET,
            REDDIT_USER_AGENT,
            num_sentences=60
        )
        all_sentences.extend(reddit_sentences)
    else:
        print("⚠️  Skipping Reddit (需要填入 API credentials)")
    
    # 從 Twitter 收集（40 個）
    if TWITTER_BEARER_TOKEN != "your_bearer_token_here":
        twitter_sentences = collect_from_twitter(
            TWITTER_BEARER_TOKEN,
            num_sentences=40
        )
        all_sentences.extend(twitter_sentences)
    else:
        print("⚠️  Skipping Twitter (需要填入 API credentials)")
    
    # 後處理
    all_sentences = post_process_sentences(all_sentences)
    
    # 儲存
    output_file = 'test_sentences_100.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_sentences, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Collected {len(all_sentences)} sentences")
    print(f"✅ Saved to: {output_file}")
    print('='*60)
    
    # 顯示一些樣本
    print("\nSample sentences:")
    for i, s in enumerate(all_sentences[:5], 1):
        print(f"{i}. {s['sentence']}")


# ============================================
# 備用方案：從 slang 資料的 example 收集
# ============================================

def collect_from_examples(data_path='/Users/bella/Documents/GitHub/Applied-NLP-project1//data/slang_clean_final.csv', num_sentences=100):
    """
    如果沒有 API，從 slang 資料的 example 欄位收集
    """
    import pandas as pd
    
    print("="*60)
    print("Collecting from Slang Examples")
    print("="*60)
    
    df = pd.read_csv(data_path)
    
    # 過濾有 example 的條目
    df_with_examples = df[df['example'].notna()].copy()
    print(f"Found {len(df_with_examples)} entries with examples")
    
    # 隨機抽樣
    sampled = df_with_examples.sample(min(num_sentences * 2, len(df_with_examples)))
    
    sentences = []
    seen = set()
    
    for _, row in sampled.iterrows():
        example = str(row['example']).strip()
        
        # 基本過濾
        if len(example) < 10 or len(example) > 100:
            continue
        
        # 切分句子
        sents = re.split(r'[.!?]+', example)
        
        for sent in sents:
            sent = sent.strip()
            
            if len(sent) < 10 or len(sent) > 100:
                continue
            
            sent_lower = sent.lower()
            if sent_lower in seen:
                continue
            
            if sent.count(' ') < 3:
                continue
            
            sentences.append({
                'sentence': sent,
                'source': 'slang_examples',
                'slang_word': row['word'],
                'length': len(sent.split())
            })
            seen.add(sent_lower)
            
            if len(sentences) >= num_sentences:
                break
        
        if len(sentences) >= num_sentences:
            break
    
    # 儲存
    output_file = 'test_sentences_from_examples.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Collected {len(sentences)} sentences")
    print(f"✅ Saved to: {output_file}")
    
    return sentences


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect test sentences')
    parser.add_argument('--mode', type=str, default='examples',
                       choices=['reddit', 'twitter', 'both', 'examples'],
                       help='Collection mode')
    args = parser.parse_args()
    
    if args.mode == 'examples':
        # 備用方案：從 slang examples 收集（不需要 API）
        sentences = collect_from_examples(num_sentences=100)
    else:
        # 需要 API credentials
        main()
