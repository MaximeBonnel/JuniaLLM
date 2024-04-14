import json

def json_to_md(input_file, output_file):
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    with open(output_file, 'w') as md_file:
        md_file.write("# Questions / Reponses sur Junia\n\n")
        for item in data:
            md_file.write("{} {}\n\n".format(item['instruction'], item['output']))

if __name__ == "__main__":
    input_file = "JuniaData.json"
    output_file = "Answers.md"
    json_to_md(input_file, output_file)
