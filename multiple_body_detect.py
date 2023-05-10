from body_detect import body_detect


from lib.get_input_files import get_input_files


input_path = "./body_pic/"
output_path = "./background_removal/"

list_files = get_input_files(input_path)

for file in list_files:
    body_detect(input_path=input_path, output_path=output_path, input_name=file)
