#!/usr/bin/python
import argparse
from apiclient.discovery import build
from apiclient.errors import HttpError
import time
import concurrent.futures
from unidecode import unidecode
from itertools import islice

parser = argparse.ArgumentParser()
parser.add_argument("api_key", help="youtube developer key. you can get one here"
                                    " https://console.developers.google.com/apis/")
parser.add_argument("query", help="search term for YT search")
parser.add_argument("out_path", help="file to put scraped comments in")
parser.add_argument('--threads', type=int, default=30,
                    help='how many threads to use for scraping (defaults to 30)')
parser.add_argument('--max_videos', type=int, default=50, help='how many videos to scrape')

args = parser.parse_args()


DEVELOPER_KEY = args.api_key
OUTPUT_PATH = args.out_path
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def extract_vide_ids(search_response):
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            yield search_result["id"]["videoId"]

def search_videos(query):
    """generator yielding video_ids for youtube videos matching given query"""
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    search_response = youtube.search().list(
        q=query,
        part="id,snippet",
        maxResults=50
    ).execute()

    for vid_id in extract_vide_ids(search_response):
        yield vid_id

    next_page_token = search_response.get('nextPageToken')
    while next_page_token:
        search_response = youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for vid_id in extract_vide_ids(search_response):
            yield vid_id


def get_page_comments(comments_getter, video_id, page_token=None, retries=5, timeout=0.5):
    """returns comments for the given video_id + token pointing to the next part
    if there is one"""
    if page_token:
        api_call = comments_getter.list(
            part='snippet',
            videoId=video_id,
            pageToken=page_token,
            textFormat='plainText')
    else:
        api_call = comments_getter.list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText')
    
    response = None
    for _ in range(retries):
        try:
            response = api_call.execute()
            break
        except HttpError as he:
            print he
            time.sleep(timeout)
    if response is None:
        return ([], None)
    
    comments = []
    for r in response['items']:
        comment = r['snippet']['topLevelComment']
        author = comment['snippet']['authorDisplayName']
        text = comment['snippet']['textDisplay']
        timestamp = comment['snippet']['publishedAt']
        video_id = comment['snippet']['videoId']
        comments.append((video_id, timestamp, author, text))
    return (comments, response.get('nextPageToken'))


def get_all_comments(video_id, time_limit=100):
    """scrapes all top level comments for a given video_id
    returns list of tuples (video_id, timestamp, author, text)"""
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    comments_getter = youtube.commentThreads()
    comments, next_page_token = get_page_comments(comments_getter, video_id)
    start = time.time()
    end = start
    i = 0
    while next_page_token and end - start < time_limit:
        i += 1
        comms, next_page_token = get_page_comments(comments_getter, video_id, next_page_token)
        comments.extend(comms)
        end = time.time()
    print 'done after %.2f seconds. Scraped %s pages, video_id %s' % ((end - start), i, video_id)
    return comments


videos = islice(search_videos(args.query), args.max_videos)
with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor, open(OUTPUT_PATH, 'wb') as out:
    all_futures = [executor.submit(get_all_comments, vid_id) for vid_id in videos]
    for future in all_futures:
        for video_id, timestamp, author, text in future.result():
            out.write("[%s]:\t%s\n" % (unidecode(author), unidecode(text)))
        out.flush()
