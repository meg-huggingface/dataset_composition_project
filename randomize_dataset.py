import sys
import random
from datasets import load_dataset, Dataset

"""
This code is designed to read in the ImageNet 1K ILSVRC dataset from the Hugging Face Hub, 
then create a new version of this dataset with {percentage} lines with random labels between 0-9,
then upload this new version of the Hugging Face Hub, in the Data Composition organization:
https://huggingface.co/datasets/datacomp
"""

# The number of examples/instances in this dataset is copied from the model card: 
# https://huggingface.co/datasets/ILSVRC/imagenet-1k
NUM_EXAMPLES = 1281167
DEV_AMOUNT = 100

def main(percentage):
  start = time.time()
  
  # Set the random seed, based on the percentage, so that our random changes are reproducible.
  random.seed(percentage)

  # Load the dataset from the HF hub. Use streaming so as not to load the entire dataset at once.
  # Use the .take(DEV_AMOUNT) to only grab a small chunk of instances to develop with.
  dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True) #.take(DEV_AMOUNT)
  
  # Create a set of indices that are randomly chosen, to change their labels.
  # Specifically, randomly choose NUM_EXAMPLES/percentage indices.
  randomize_subset = set(random.sample(range(0, NUM_EXAMPLES), round(NUM_EXAMPLES/float(percentage))))

  # Update the dataset so that the labels are randomized
  updated_dataset = dataset.map(randomize_labels, with_indices=True,
                                features=dataset.features, batched=True)

  # Upload the new version of the dataset (this will take awhile)
  Dataset.from_generator(updated_dataset.__iter__).push_to_hub("datacomp/imagenet-1k-random" + str(percentage))
  
  end = time.time()
  print("That took %d seconds" % (end - start))

def randomize_labels(examples, indices):
  # What set of examples should be randomized in this batch?
  # This is the intersection of the batch indices and the indices we randomly selected to change the labels of.
  batch_subset = list(set(indices) & randomize_subset)
  # If this batch has indices that we're changing the label of....
  if batch_subset != []:
    # Change the label to a random integer between 0 and 9
    for n in range(len(indices)):
        index = indices[n]
        examples["label"][n] = random.randint(0, 9) if index in batch_subset else examples["label"][n]
return examples


if __name__ == "__main__":
  if len(sys.argv) > 1:
    percentage = float(sys.argv[1])
  else:
    percentage = 10
  main(percentage)
