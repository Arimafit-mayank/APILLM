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

@app.route('/AITrainer', methods = ['POST'])
def AITrainer(data):
    data = request.json
    goal = data["Fitness_Goals"]
    intensity = data["I_Workout"]
    workout_type = data["Workout_Type"]
    weight = data["weight"]
    preference = data["Preferences"]["Selected_Options"]
    print(preference)
    sample='''Name of Week day: Muscle Group1
    4 exercises in bullet points
    Name of Week day: Muscle Group2
    4 exercises in bullet points
    Name of Week day: Muscle Group3
    4 exercises in bullet points
    Name of Week day: Muscle Group4
    4 exercises in bullet points

    '''
    chain = get_chain_workout()
    query = f"""
    Please create a one-week workout plan. 
    Make sure each muscle group (chest, back, arms, legs, abs) is trained.
    Use only the equipment I have: {preference}. 
    My workout type is: {workout_type}.
    My current weight is: {weight}. 
    My goal is: {goal}. 
    My workout intensity is: {intensity}.
    Include rest days for days I don't workout.
    Display all 7 days with at least 4 exercises per day.
    Provide proper reps and sets.
    Do not use any special characters or symbols in the response.
    Only use simple text and the English alphabet.
    The format should be: ${sample}.
    """
    response1 = chain.invoke(query)["result"]

    # query = f"Give the output in this format: ${sample}. Please Don't use any special character in the response. Don't use | symbol. Give me neat and clean result. Write only simple text. Important Please don't use any special character and write only simple text response with english alphabet only. Create one week workout plan which ensures that each muscle group that is chest, back, arms, legs, abs are trained. I have these equipments and these are my workout days {preference}. Only give those workouts that can be done with the equipments I mentioned and I want {workout_type}. My current weight is {weight} and my goal is {goal} and I wworkout {intensity}. Make sure to include the rest days as the days which are not my workout days and display all the 7 days with atleast 4 exercises for each day with proper reps and sets."
    # response1 = chain.invoke(query)["result"]
    # print("Week 1: ", response1)
    response2 = chain.invoke(f"""
Keep the exact same format as this: ${response1}. 
Do not add anything extra in the response. 
This was my first week workout plan: {response1}. 
Create the next week's workout plan. 
Please make sure to change the exercises and increase the intensity by recommending more reps or increasing weights.
""")["result"]

    response3 = chain.invoke(f"""
    Keep the exact same format as this: ${response2}. 
    Do not add anything extra in the response. 
    This was my second week workout plan: {response2}. 
    Create the next week's workout plan. 
    Please make sure to change the exercises and increase the intensity by recommending more reps or increasing weights.
    """)["result"]

    response4 = chain.invoke(f"""
    Keep the exact same format as this: ${response3}. 
    Do not add anything extra in the response. 
    This was my third week workout plan: {response3}. 
    Create the next week's workout plan. 
    Please make sure to change the exercises and increase the intensity by recommending more reps or increasing weights.
    """)["result"]
    response = "Week1: " + response1 + "\nWeek2: " + response2 +"\nWeek3: " + response3 +"\nWeek4: " + response4
    return response


@app.route('/get_data', methods=['GET', 'POST'])
def handle_data():
    user = request.json  # Extract JSON data from the request
    weight = user["data"][0]["actualWeight"]
    target_weight= user["data"][0]["targetWeight"]
    age = user["data"][0]["age"]
    height = user["data"][0]["height"]
    if user["data"][0]["gender"] == "Male":
        BMR = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        BMR = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    # print(BMR)
    if user["data"][0]["activityLevel"] == "sedentary":
        Calorie_Calculation = BMR * 1.2
    elif user["data"][0]["activityLevel"] == "lightly active":
        Calorie_Calculation = BMR * 1.375
    else:
        Calorie_Calculation = BMR * 1.55

    count_cal = 0
    sm_cal=0
    count_water = 0
    sm_water = 0
    count_steps = 0
    sm_steps = 0
    count_sleep = 0
    sm_sleep = 0

    for i in range(7):
        if user["data"][0]["calories"]!=0:
            count_cal+=1
            sm_cal+=user["data"][0]["calories"]
        if user["data"][0]["water"]!=0:
            count_water+=1
            sm_water+=user["data"][0]["water"]*0.25
        if user["data"][0]["steps"]!=0:
            count_steps+=1
            sm_steps+=user["data"][0]["steps"]
        if user["data"][0]["sleep"]!=0:
            count_sleep+=1
            sm_sleep+=user["data"][0]["sleep"]

    avg_calories = sm_cal/count_cal
    avg_water = sm_water/count_water
    avg_steps = sm_steps/count_steps
    avg_sleep = sm_sleep/count_sleep


    if age>=9 and age<=13:
        water_intake = 1.89
    elif age>=14 and age<=18:
        water_intake = 2.6
    else:
        water_intake = 3.5

    if target_weight<weight:
        steps_taken = 10000
        Calorie_Calculation-=400
    elif target_weight==weight:
        steps_taken = 8000
    else:
        steps_taken = 7500
        Calorie_Calculation+=400

    # print(avg_calories,avg_sleep,avg_steps,avg_water)

    # print(Calorie_Calculation,"6 hours to 8 hours", steps_taken, water_intake)
    response = llm.invoke(f"Give me insights for these: My 7 days average data is as follows: Average Calories taken ${avg_calories}, Avergae sleep taken ${avg_sleep}, Average steps taken ${avg_steps}, Average water intake ${avg_water} litres and Recommended data to achieve my goal for a day is as follows:/n  Calories taken ${Calorie_Calculation}, Sleep to be taken 6hours to 8 hours, Steps to be taken ${steps_taken}, Water intaken should be ${water_intake} litres. Provide me insights for each data what things i need to improve in each data. Give me an overall rating as good, better , best, bad , very bad. Use a natural tone and provide motivation to do things.")
    return jsonify({"repsonse":response})
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
    app.run()
