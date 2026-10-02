import numpy as np
import pandas as pd

# Note - study hour time = 1-10 hrs 
study_hours = np.random.randint(1, 11, 100)  
print(study_hours)   

# we want attendance range as = 50-100
attendance = np.random.randint(50, 101, 100)
print(attendance)

# previous_marks range - [30-95]
previous_marks = np.random.randint(30, 96, 100)
print(previous_marks)

# assignment_score range - [30-100]
assignment_score = np.random.randint(30, 101, 100)
print(assignment_score)

# test_score range - [30-100]
test_score = np.random.randint(30, 101, 100)
print(test_score)

# yaha sare features 100 tk range kr rhe h except study_hours 
# therefore, hm use skbe barabar wali scale pr laenge as
# 10 hrs -> ~100    5 hrs -> 50     1 hr -> 10      islie hmne 10 s multiply kia h
final_marks = (
    (study_hours * 10) * 0.20
    + attendance * 0.15
    + previous_marks * 0.25
    + assignment_score * 0.15
    + test_score * 0.25
)

print(final_marks)


# Creating dataframes

data= {
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks,
    "assignment_score": assignment_score,
    "test_score": test_score,
    "final_marks": final_marks,
}

df = pd.DataFrame(data)

print(df)

df.to_csv("data/student_data.csv", index=False)