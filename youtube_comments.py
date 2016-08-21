#!/usr/bin/python
import argparse
from apiclient.discovery import build
from apiclient.errors import HttpError
import time
import concurrent.futures
from unidecode import unidecode

parser = argparse.ArgumentParser()
parser.add_argument("api_key", help="youtube developer key")
parser.add_argument("query", help="what videos to look for")
parser.add_argument("out_path", help="where to put the comments")
args = parser.parse_args()


DEVELOPER_KEY = args.api_key
OUTPUT_PATH = args.out_path
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def search_videos(query):
    """returns video_ids for youtube videos matching the query"""
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    search_response = youtube.search().list(
        q=query,
        part="id,snippet",
        maxResults=50
    ).execute()
    
    videos = []
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            videos.append(search_result["id"]["videoId"])
    return videos


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


def get_all_comments(video_id, time_limit=10):
    """scrapes all top level comments for a given video_id
    returns list of tuples (video_id, timestamp, author, text)"""
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    comments_getter = youtube.commentThreads()
    comments, next_page_token = get_page_comments(comments_getter, video_id)
    start = time.time()
    end = start
    while next_page_token and end - start < time_limit:
        comms, next_page_token = get_page_comments(comments_getter, video_id, next_page_token)
        comments.extend(comms)
        end = time.time()
    return comments


videos = search_videos(args.query)
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    all_futures = [executor.submit(get_all_comments, vid_id) for vid_id in videos[:2]]
    all_comments = [comment for future in all_futures for comment in future.result()]
    
with open(OUTPUT_PATH, 'wb') as out:
    for video_id, timestamp, author, text in all_comments:
        out.write("%s:\t%s\n" % (unidecode(author), unidecode(text)))