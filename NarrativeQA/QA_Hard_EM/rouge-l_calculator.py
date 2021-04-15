from rouge import Rouge

first_answers = [
    "he is a high school student in Phoenix.",
    "it takes place in Mark's parents basement.",
    "mark talks about what goes on at school and in the community.",
    "malcolm.",
    "she jams her medals and accolades.",
    "he dismantles it and attaches it to his mother's jeep.",
    "he tells them to make their own future.",
    "mark and nora.",
    "it causes trouble.",
    "parent's basement",
    "his radio station",
    "everybody knows",
    "fellow student",
    "malcom's suicide",
    "do something about their problems.",
    "the fcc",
    "to retain government funding",
    "expelled them",
    "nora",
    "phoenix, arizona.",
    "in his parent's basement.",
    "his radio stations theme song.",
    "commits suicide.",
    "puts them in the microwave.",
    "when the microwave explodes.",
    "to investigate the radio show.",
    "expelling them.",
    "the cops and the fcc.",
    "protesting."
]

second_answers = [
    "a loner and outsider student with a radio station.",
    "phoenix, arizona",
    "because he has a thing to say about what is happening at his school and the community.",
    "malcolm.",
    "her award medals",
    "dismantle it.",
    "that they should make their own future because the world belongs to them.",
    "mark and nora.",
    "it causes much trouble in the community.",
    "at the basement of his home",
    "his unauthorized radio station.",
    "everybody know's",
    "a fellow student",
    "to confront him after malcolm commits suicide.",
    "to do something about their problems instead of committing suicide.",
    "fcc",
    "for government funding",
    "expell them",
    "nora",
    "phoenix, arizona.",
    "his parents' basement.",
    "it is the theme song.",
    "commits suicide.",
    "melts them in a microwave.",
    "microwaving her medals",
    "because of trouble caused by the radio station.",
    "expelling the students",
    "the police and the fcc.",
    "protesting"
]

predictions = [
    "a high school student",
    "phoenix , arizona",
    "to hear his show",
    "malcolm",
    "jams her various medals and accolades",
    "starts an fm pirate",
    "that the world belongs to them",
    "mark and nora",
    "causes so much trouble in the community",
    "his parents ' house",
    "his unauthorized radio station",
    "everybody knows by leonard cohen",
    "fellow student",
    "a student named malcolm commits suicide",
    "do something about their problems",
    "the fcc",
    "to retain government funding",
    "expelling problem students",
    "nora",
    "phoenix , arizona",
    "his parents ' house",
    "leonard cohen",
    "commits suicide",
    "jams her various medals and accolades into a microwave",
    "the microwave explodes",
    "to investigate",
    "below-average standardized test scores",
    "the police and the fcc",
    "protesting"
]

# Improve calculation making lowercase all letters of answers
for i, answer in enumerate(first_answers):
    first_answers[i] = answer.casefold()

for i, answer in enumerate(second_answers):
    second_answers[i] = answer.casefold()

# Improve predictions making lowercase all letters and deleting spaces before punctuation and apostrophe
for i, prediction in enumerate(predictions):
    prediction = prediction.casefold()
    prediction = prediction.replace(" ,", ",")
    prediction = prediction.replace(" ;", ";")
    prediction = prediction.replace(" .", ".")
    predictions[i] = prediction.replace(" '", "'")

rouge = Rouge()
first_scores = rouge.get_scores(first_answers, predictions)
second_scores = rouge.get_scores(second_answers, predictions)

print("First answers ROUGE-L:")
for i in range(len(first_scores)):
    print(str(i+1) + ". " + str(first_scores[i]["rouge-l"]["f"] * 100))

print("\nSecond answers ROUGE-L:")
for i in range(len(second_scores)):
    print(str(i+1) + ". " + str(second_scores[i]["rouge-l"]["f"] * 100))