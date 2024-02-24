from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["API_KEY"], temperature=0.7)


instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

vector_file_path_exercise = "faiss_index_exercise"
vector_file_path_diet = "faiss_index_diet"

def create_workout_vectordb():
    loader = CSVLoader("Workout.csv", encoding = "utf8")
    workout = loader.load()
    workout_vector_db = FAISS.from_documents(documents=workout,
                                 embedding=instructor_embeddings)
    workout_vector_db.save_local(vector_file_path_exercise)
    

def create_diet_vectordb():
    loader = CSVLoader("Diet.csv")
    diet = loader.load()
    diet_vector_db = FAISS.from_documents(documents=diet,
                                 embedding=instructor_embeddings)
    diet_vector_db.save_local(vector_file_path_diet)

def get_chain_workout():
    exercise_db = FAISS.load_local(vector_file_path_exercise, instructor_embeddings)
    retriever = exercise_db.as_retriever(score_threshold = 0.7)
    workout_chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True
                            )
    return workout_chain

def get_chain_diet():
    diet_db = FAISS.load_local(vector_file_path_diet, instructor_embeddings)
    retriever = diet_db.as_retriever(score_threshold = 0.7)
    diet_chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True
                            )
    return diet_chain



@app.route('/request', methods = ['POST'])
def main():
    print("Select your Choice")
    print("Exercise\n Diet")
    choice = request.form.get('choice')
    
    if choice =='Exercise':
        print("Ask anything about exercises, I can create custom workout plans for you")
        query = request.form.get('query')
        #return jsonify({'query':query})
        chain = get_chain_workout()
        
    else:
        print("Ask anything about diet plan, I can create custom diet plans fro you")
        query = request.form.get('query')
        chain = get_chain_diet()

    response = chain.invoke(query)["result"]
    return jsonify({'response':response})

if __name__ == '__main__':
    app.run(debug=True)



