from flask import Flask 

app = Flask(_name_) 

@app.route('/post/<int:id>') 
def show_post(id): 
	# Shows the post with given id. 
	return f'This post has the id {id}'

@app.route('/user/<username>') 
def show_user(username): 
	# Greet the user 
	return f'Hello {username} !'

# Pass the required route to the decorator. 
@app.route("/hello") 
def hello(): 
	return "Hello, Welcome to GeeksForGeeks"
	
@app.route("/") 
def interface(): 
	return "Homepage of GeeksForGeeks"

if _name_ == "_main_": 
	app.run(debug=True)
	

    from flask import Flask, request, jsonify
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer

app = Flask(_name_)

# Load pre-trained DistilBERT model and tokenizer
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

@app.route('/answer', methods=['POST'])
def answer_question():
    # Retrieve data from request
    data = request.get_json()
    context = data['context']
    question = data['question']

    # Call function to answer the question
    answer = answer_question_helper(question, context)

    # Return the answer as JSON response
    return jsonify({'answer': answer})

def answer_question_helper(question, context):
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    start_logits, end_logits = model(*inputs).start_logits, model(*inputs).end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))
    return answer

if _name_ == '_main_':
    app.run(debug=True)
	




    from flask import Blueprint, request, jsonify, render_template
from main.qa_utils import answer_question

main = Blueprint('main', _name_)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/answer', methods=['POST'])
def answer():
    data = request.get_json()
    context = data['context']
    question = data['question']
    answer = answer_question(question, context)
    return jsonify({'answer': answer})