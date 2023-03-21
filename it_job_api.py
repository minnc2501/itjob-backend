import pandas as pd
import numpy as np
import RAKE
import operator
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import uvicorn
from fastapi.responses import JSONResponse, UJSONResponse, FileResponse
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

origins = [    
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/api/job_it')
def api_job_it(
    input: str
    ):

    job_postings = 'dice_com-job_us_sample.csv'
    df_job = pd.read_csv(job_postings)
    # print(df_job)

    df_job.drop('postdate', inplace=True, axis = 1)
    df_job.drop('shift', inplace=True, axis = 1)
    df_job.drop('site_name', inplace=True, axis = 1)
    df_job.drop('uniq_id', inplace=True, axis = 1)
    df_job.drop('jobid', inplace=True, axis = 1)

    df_job.drop_duplicates()
    df_job.dropna(inplace=True)

    jobsNull = df_job[df_job['skills'] == "Null"].index
    jobDesc1 = df_job[df_job['skills'] == "Please see job description"].index
    jobDesc2 = df_job[df_job['skills'] == "(See Job Description)"].index
    jobDesc3 = df_job[df_job['skills'] == "SEE BELOW"].index
    jobDesc4 = df_job[df_job['skills'] == "Telecommuting not available Travel not required"].index
    jobDesc5 = df_job[df_job['skills'] == "Refer to Job Description"].index
    jobDesc6 = df_job[df_job['skills'] == "Please see Required Skills"].index

    df_job.drop(jobsNull, inplace=True)
    df_job.drop(jobDesc1, inplace=True)
    df_job.drop(jobDesc2, inplace=True)
    df_job.drop(jobDesc3, inplace=True)
    df_job.drop(jobDesc4, inplace=True)
    df_job.drop(jobDesc5, inplace=True)
    df_job.drop(jobDesc6, inplace=True)

    #CSV to TXT files

    # print(df_job.info())
    job = []
    stopwordsList = []
    cleanJobs = []

    #Get the stopwords and store in list
    with open('stopwords.txt', 'r', encoding= 'utf-8') as f:
        for word in f:
            word = word.split('\n') 
            stopwordsList.append(word[0])

    #Tokenzing and Removing stopswords from jobtitle
    # print(stopwordsList)
    #Convert all words to lower case and change the shortfrom
    for i in df_job['jobtitle'].values:
        jobs = i.lower()
        jobs = jobs.replace('QA', 'Quality Assurance')
        jobs = jobs.replace('sr', 'Senior')
        jobs = jobs.replace('jr', 'Junior')
        jobs = jobs.replace('qm', 'Quality Manager')
        job.append(jobs)
    #Tokenzing and Removing stopswords from stop words
    for j in job: 
        text_tokens = word_tokenize(j)
        tokens_without_sw = [f for f in text_tokens if not f in stopwordsList]
        cleanJobs.append(' '.join(tokens_without_sw))
    df_job['clean_jobtitle'] = cleanJobs
    # qty = df_job['clean_jobtitle'].value_counts()[:5].tolist()
    label = df_job['clean_jobtitle'].value_counts()[:5].index.tolist()      
    # print(qty)
    # print('test: ' + str(label))

    #Skills

    skillsTokenized = []
    stopwordsSkills = []

    #Get the stopwords and store in list
    with open('stopwordsskills.txt', 'r', encoding= 'utf-8') as f:
        for word in f:
            word = word.split('\n') 
            stopwordsSkills.append(word[0])

    for k in df_job['skills'].values:
        k = str(k).split(', ')
        skillstoken_without_sw = [f for f in k if not f.lower() in stopwordsSkills]
        for j in skillstoken_without_sw:
            skillsTokenized.append(j)

    df = pd.DataFrame({'skills': skillsTokenized})
    # qtySkills = df['skills'].value_counts().tolist()
    # labelSkills = df['skills'].value_counts().index.tolist()
    # print('test \n' + str(df['skills'].value_counts()[:5]))

    #data mining
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_job['clean_jobtitle'].values)
    # print(X.shape)
    # analyze = vectorizer.build_analyzer()
    features =  vectorizer.get_feature_names_out()

    #Clustering using KMeans
    model = KMeans(n_clusters= 7, init='k-means++', max_iter = 600, n_init =1, random_state = 42)
    pred = model.fit_predict(X)
    # order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names_out()

    sklearn_pca = PCA(n_components= 2)
    Y_sklearn = sklearn_pca.fit_transform(X.toarray())

    kmeans = KMeans(n_clusters= 7, init='k-means++', max_iter = 600, n_init =1, random_state = 42)
    fitted = kmeans.fit(Y_sklearn)
    prediction = kmeans.predict(Y_sklearn)
    center = kmeans.cluster_centers_

    # from sklearn.metrics import silhouette_score
    # print('test {}'.format(silhouette_score(X, model.labels_, metric = 'euclidean')) )

    # def get_top_keywords(data, clusters, labels, n_terms):
    #     df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    #     for i,r in df.iterrows():
    #         print('\nCluster {}'.format(i))
    #         print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
    # get_top_keywords(X,pred, features, 10)

    label = []
    for i in df_job['clean_jobtitle'].values:
        vec = vectorizer.transform([i])
        pred = model.predict(vec)
        if pred == 0:
            label.append('Business Solution Consultant')
        elif pred == 1:
            label.append('Frontend/Backend/FullStack')
        elif pred == 2:
            label.append('Analyst')
        elif pred == 3:
            label.append('Project Management')
        elif pred == 4:
            label.append('DevOps/Software Engineer')
        elif pred == 5:
            label.append('IT Business Management')
        elif pred == 6:
            label.append('Cloud Architect/Network Systems')
    df_job['Label'] = label

    jobSkills = []
    for i in df_job['skills']:
        jobSkills.append(i.lower())

    Xclass = vectorizer.fit_transform(jobSkills)
    X_train, X_test, y_train, y_test = train_test_split(Xclass, label, test_size = 0.2, random_state =42)

    svmmodel = svm.SVC(C = 5, gamma = 1, kernel = 'rbf', probability = True)
    svmfit = svmmodel.fit(X_train, y_train)
    svm_predictions = svmfit.predict(X_test)
    # svm_acc = accuracy_score(y_test, svm_predictions)
    # print(str(svm_acc))

    # Data analysis
    # labelData = df_job[df_job['Label'] == 'Frontend/Backend/FullStack' ]
    # skillsClass = []

    # for index, row in labelData.iterrows():
    #     skills = [row['skills']]
    #     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    #     for j in skillstokens_without_sw:
    #         skillsClass.append(j)
    # df_frontend = pd.DataFrame({'skills': skillsClass})
    # qtySkills = df_frontend['skills'].value_counts().to_list()
    # labelSkills = df_frontend['skills'].value_counts().index.to_list()

    # labelData = df_job[df_job['Label'] == 'Analyst' ]
    # skillsClass = []
    # for index, row in labelData.iterrows():
    #     skills = [row['skills']]
    #     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    #     for j in skillstokens_without_sw:
    #         skillsClass.append(j)
    # df_analyst = pd.DataFrame({'skills': skillsClass})
    # qtySkills = df_analyst['skills'].value_counts().to_list()
    # labelSkills = df_analyst['skills'].value_counts().index.to_list()

    # labelData = df_job[df_job['Label'] == 'Business Solution Consultant' ]
    # skillsClass = []
    # for index, row in labelData.iterrows():
    #     skills = [row['skills']]
    #     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    #     for j in skillstokens_without_sw:
    #         skillsClass.append(j)
    # df_consultant = pd.DataFrame({'skills': skillsClass})
    # qtySkills = df_consultant['skills'].value_counts().to_list()
    # labelSkills = df_consultant['skills'].value_counts().index.to_list()

    # labelData = df_job[df_job['Label'] == 'Cloud Architect/Network Systems' ]
    # skillsClass = []
    # for index, row in labelData.iterrows():
    #     skills = [row['skills']]
    #     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    #     for j in skillstokens_without_sw:
    #         skillsClass.append(j)
    # df_cloud = pd.DataFrame({'skills': skillsClass})
    # qtySkills = df_cloud['skills'].value_counts().to_list()
    # labelSkills = df_cloud['skills'].value_counts().index.to_list()

    # labelData = df_job[df_job['Label'] == 'Analyst' ]
    # skillsClass = []
    # for index, row in labelData.iterrows():
    #     skills = [row['skills']]
    #     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    #     for j in skillstokens_without_sw:
    #         skillsClass.append(j)
    # df_analyst = pd.DataFrame({'skills': skillsClass})
    # qtySkills = df_analyst['skills'].value_counts().to_list()
    # labelSkills = df_analyst['skills'].value_counts().index.to_list()

    # labelData = df_job[df_job['Label'] == 'DevOps/Software Engineer' ]
    # skillsClass = []
    # for index, row in labelData.iterrows():
    #     skills = [row['skills']]
    #     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    #     for j in skillstokens_without_sw:
    #         skillsClass.append(j)
    # df_engineer = pd.DataFrame({'skills': skillsClass})
    # qtySkills = df_engineer['skills'].value_counts().to_list()
    # labelSkills = df_engineer['skills'].value_counts().index.to_list()

    # labelData = df_job[df_job['Label'] == 'IT Business Management' ]
    # skillsClass = []
    # for index, row in labelData.iterrows():
    #     skills = [row['skills']]
    #     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    #     for j in skillstokens_without_sw:
    #         skillsClass.append(j)
    # df_bus = pd.DataFrame({'skills': skillsClass})
    # qtySkills = df_bus['skills'].value_counts().to_list()
    # labelSkills = df_bus['skills'].value_counts().index.to_list()

    # labelData = df_job[df_job['Label'] == 'Project Management' ]
    # skillsClass = []
    # for index, row in labelData.iterrows():
    #     skills = [row['skills']]
    #     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    #     for j in skillstokens_without_sw:
    #         skillsClass.append(j)
    # df_project = pd.DataFrame({'skills': skillsClass})
    # qtySkills = df_project['skills'].value_counts().to_list()
    # labelSkills = df_project['skills'].value_counts().index.to_list()

    userInput = input
    pred = vectorizer.transform([userInput.lower()])

    output = svmmodel.predict(pred)

    cos = []
    labelData = df_job[df_job['Label'] == output[0]]

    for index, row in labelData.iterrows():
        skills = [row['skills']]
        sillVec = vectorizer.transform(skills)
        cos_lib = cosine_similarity(sillVec, pred)
        cos.append(cos_lib[0][0])
    labelData['cosine_similarity'] = cos

    df_result = labelData.sort_values('cosine_similarity', ascending=False)[['advertiserurl', 'company', 'employmenttype_jobstatus', 'jobdescription', 'joblocation_address','jobtitle','skills', 'Label']]
    top_10 = df_result.head(10)
    response_json = top_10.to_dict('records')
    # print(top_10)
    return response_json

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        )