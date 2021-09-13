#!/usr/bin/env bash

trap "exit" INT
for vector_path in "$@"
do
  for method in subjective snli relation sentence_sentiment document_sentiment
  do
    exeval --log --vector_path $vector_path $method
  done

  for task in pos ner chunk
  do
    exeval --log --vector_path $vector_path sequence_labeling --subtask $task
  done
done
