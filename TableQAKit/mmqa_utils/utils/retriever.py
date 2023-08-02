import json
import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
DATASET_SPLIT = 'dev'
TYPE_FILE = os.path.join(ROOT_DIR, f"utils/retriever_files/question_type_{DATASET_SPLIT}.json")
QIM_FILE = os.path.join(ROOT_DIR, f"utils/retriever_files/question_image_match_scores_{DATASET_SPLIT}.json")
QPM_FILE = os.path.join(ROOT_DIR, f"utils/retriever_files/question_passage_match_scores_{DATASET_SPLIT}.json")

"""
Match Score Files be like:
{
    "eid": {
        "doc_id1": score,
        "doc_id2": score,
    }
}
"""


class Retriever:
    def __init__(self, type_file = TYPE_FILE, qim_file = QIM_FILE, qpm_file = QPM_FILE):
        self.type_dict = json.load(open(type_file, 'r'))
        self.qim_dict = json.load(open(qim_file, 'r'))
        self.qpm_dict = json.load(open(qpm_file, 'r'))
    
    def retrieve(self, g_data_item, eid = None, qid = None):
        if eid is None or qid is None:
            raise ValueError("eid or qid is None")

        qtype = self.type_dict[qid]['predict']
        # Get the retrieved passages
        new_passages = {"id": [], "title": [], "text": [], "url": []}
        r_passage_ids = []
        passage_scores_dict = self.qpm_dict[str(eid)].copy()
        # get top3
        if len(passage_scores_dict) > 3:
            passage_scores_dict = dict(sorted(passage_scores_dict.items(), key=lambda x:x[1], reverse=True)[:3])

        for _id, title, text, url in zip(g_data_item['passages']['id'],
                                            g_data_item['passages']['title'],
                                            g_data_item['passages']['text'],
                                            g_data_item['passages']['url']):
                if _id in r_passage_ids:
                    continue
                r_passage_ids.append(_id)
                flag = True
                # TODO: decide when set flag to False
                if not (_id in passage_scores_dict.keys()):
                    flag = False

                if flag:
                    new_passages["id"].append(_id)
                    new_passages["title"].append(title)
                    new_passages["text"].append(text)
                    new_passages["url"].append(url)
        
        # Get the retrieved images
        new_images = {"id": [], "title": [],"pic": [], "url": [], "path": []}
        r_image_ids = []
        image_scores_dict = self.qim_dict[str(eid)].copy()
        # get top3
        if len(image_scores_dict) > 3:
            image_scores_dict = dict(sorted(image_scores_dict.items(), key=lambda x:x[1], reverse=True)[:3])

        for _id, title, pic, url, path in zip(g_data_item['images']['id'],
                                                g_data_item['images']['title'],
                                                g_data_item['images']['pic'],
                                                g_data_item['images']['url'],
                                                g_data_item['images']['path']):
                if _id in r_image_ids:
                    continue
                r_image_ids.append(_id)
                flag = True
                # TODO: decide when set flag to False
                if not (_id in image_scores_dict.keys()):
                    flag = False

                if flag:
                    new_images["id"].append(_id)
                    new_images["title"].append(title)
                    new_images["pic"].append(pic)
                    new_images["url"].append(url)
                    new_images["path"].append(path)

        return new_passages, new_images, qtype

if __name__ == '__main__':
    pass