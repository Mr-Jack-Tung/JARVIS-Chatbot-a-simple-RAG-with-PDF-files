# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# File: parse_response_o1.py
# Date: 18 December 2024 - 09 PM

import re

# https://huggingface.co/spaces/sambanovasystems/Llama3.1-Instruct-O1/blob/main/app.py
def parse_response(response):
	"""Parses the response from the API."""
	answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
	reflection_match = re.search(r'<reflection>(.*?)</reflection>', response, re.DOTALL)

	answer = answer_match.group(1).strip() if answer_match else ""
	reflection = reflection_match.group(1).strip() if reflection_match else ""
	steps = re.findall(r'<step>(.*?)</step>', response, re.DOTALL)

	if answer == "":
		return response #, "", ""

	# return answer, reflection, steps

	output_string = ""
	if steps:
		output_string += "\n\n**Steps:**"
		for i, step in enumerate(steps):
			output_string += "\n"+ str(i) + ": " + step
	if reflection:
		output_string += "\n\n**Reflection:** " + reflection
	if answer:
		output_string += "\n\n**Answer:** " + answer + "\n"

	return output_string