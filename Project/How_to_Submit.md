# ✅ **How Students Should Submit Their CNN Projects**

a successful submission mechanism should accommodate:

1. **Code and Model Files:** A system capable of handling zipped folders or repository links containing all scripts (
   e.g., Python/Jupyter Notebooks) and the final, best-performing trained model file (saved using ModelCheckpoint).
2. **Documentation/Report:** A mechanism for submitting a written document detailing the steps and results (PDF format
   is common).
3. **Data Documentation (if applicable):** Documentation regarding custom datasets collected, including data source,
   image capture conditions, and class definitions.

### List of Items to Be Delivered (Deliverables)

Students should deliver a complete package demonstrating their ability to execute and document the project methodology.

| Category                             | Deliverable Item                  | Supporting Source Content                                                                                                                                                                    |
|:-------------------------------------|:----------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **I. Code & Model**                  | **Trained CNN Model**             | The best-performing model must be saved (e.g., using ModelCheckpoint).                                                                                                                       |
|                                      | **Source Code/Scripts**           | Code must include the implementation of the CNN architecture, data preprocessing steps, and the training loop.                                                                               |
|                                      | **Evaluation Script Output**      | A script or log file showing the raw output of the classification report, including overall accuracy and F1-scores on the test set.                                                          |
| **II. Written Report/Documentation** | **Introduction/Background**       | Explanation of the project title and task (e.g., Road Surface Condition Classification or Fish Species Recognition).                                                                         |
|                                      | **Significance and Applications** | Description of the real-world value of the detection system (e.g., preventing accidents, monitoring biodiversity, smart city monitoring).                                                    |
|                                      | **Dataset Documentation**         | Description of the dataset source, image capture conditions, and the data split used (typically 70% Training, 15% Validation, 15% Test).                                                     |
|                                      | **Methodology**                   | Detailed explanation of the CNN architecture used, data preprocessing steps (resizing, normalization, augmentation), and training parameters (optimizer, loss function, batch size, epochs). |
|                                      | **Evaluation Results**            | Presentation and discussion of the **overall accuracy**, class-wise **F1-scores**, and the **confusion matrix**. ROC curves may also be included for binary classification tasks.            |
|                                      | **Discussion/Challenges**         | Analysis of common challenges faced (e.g., lighting variability, occlusions, limited dataset size) and potential improvements considered (e.g., transfer learning or segmentation models).   |
|                                      | **References**                    | List of all datasets and research papers cited.                                                                                                                                              |

Below is a **standard submission package** that students must follow.
This ensures organization, readability, and fair grading.

---

# **📦 1. Submission Folder Structure**

Students must submit a folder named:

```
Project_<Number>_<TeamNumber>/
```

Inside it, the structure should be:

```
Project_<Number>_<TeamNumber>/
│
├── report.pdf                 → full project report
├── code/
│   ├── train.py               → training script
│   ├── model.py               → model architecture
│   ├── dataset.py             → dataset loading & augmentation
│   ├── evaluate.py            → evaluation script
│   └── utils.py               → helper functions (optional)
│
├── saved_model/
│   └── best_model.h5          → final trained model
│
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
└── README.md                  → instructions on how to run code
```

---

# **📄 2. The Report (PDF) Must Include**

A **4–8 page** report with the following sections:

1. **Title page**

    * Project name
    * Student names & IDs
    * Submission date
    * Course Name & Instructor Names

2. **Introduction**

    * What is the problem?
    * Why is it important?

3. **Dataset**

    * Source of dataset
    * Number of classes
    * Number of images per class
    * Preprocessing steps

4. **Methodology**

    * Model architecture (diagram or summary)
    * Data augmentation
    * Training procedure
    * Hyperparameters used

5. **Results**

    * Accuracy
    * Loss curves
    * Confusion matrix
    * F1-scores

6. **Discussion**

    * What worked well
    * What failed and why
    * Limitations

7. **Conclusion & Future Work**

8. **References**

---

# **🧪 3. Code Requirements**

Students must:

* Keep scripts clean and organized
* Comment important lines
* Avoid uploading a huge dataset (point to source instead)
* Make sure the code runs end-to-end
* Use a **requirements.txt** file, e.g.:

```
tensorflow
numpy
matplotlib
scikit-learn
pillow
```

---

# **▶ 4. Demonstration Video (Optional but Highly Recommended)**

Students record a **3–7 minute** video:

1. Introduce the problem
2. Show model architecture
3. Run the code
4. Show results
5. Brief conclusion

They upload the video to Google Drive / YouTube (unlisted) and include the link in the report.

---

# **📝 5. README.md File**

The README must include:

* Project description
* Dataset link
* How to install dependencies
* How to run the training script
* How to run evaluation
* How to load the saved model for inference

Example:

```
python code/train.py
python code/evaluate.py
```

---

# **🗂 6. Allowed Submission Methods**

Students can submit via:

### ✔ Email

### ✔ Google Drive folder link

### ✔ GitHub repository (best option)

If using GitHub, ensure:

* `README.md` is complete
* `saved_model/` contains the final model or link to download
* Code runs without errors

---

# **📌 7. Grading Rubric**

You can evaluate submissions like this:

| Component                                  | Weight |
|--------------------------------------------|--------|
| Report quality                             | 30%    |
| Code correctness & organization            | 25%    |
| Model performance                          | 20%    |
| Visualizations (curves & confusion matrix) | 10%    |
| Originality / effort                       | 10%    |
| README & folder structure                  | 5%     |

***
