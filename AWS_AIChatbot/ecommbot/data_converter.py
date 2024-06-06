import pandas as pd
from langchain_core.documents import Document

def dataconverter():
    #data = pd.read_csv(r"C:\Users\raphael\robot-projet\IA-project-youtube\AWS_AIChatbot\data\flipkart_product_review.csv")
    data = pd.read_csv(r"../data/flipkart_product_review.csv")

    data = data[["product_title","review"]]
    product_list = []

    #iterate over the rowq of the dataframe
    for index, row in data.iterrows():
        #construct an object with 'product name' and 'review'
        obj = {
            'product_name' : row["product_title"],
            'review' : row["review"]
        }
        #append the object
        product_list.append(obj)

    docs = []
    for entry in product_list:
        metadata = {"product_name": entry["product_name"]}
        doc = Document(page_content=entry['review'], metadata =metadata)
        docs.append(doc)

    return docs