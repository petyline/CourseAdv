import csv
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from flask import Flask, render_template, request

app = Flask(__name__)

# Load knowledge base from CSV
def load_knowledge_base(filename):
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        knowledge_base = []
        for row in reader:
            entry = {headers[i].strip(): row[i].strip() for i in range(len(headers))}
            knowledge_base.append(entry)
    return knowledge_base


course_credit_units = {
    "MTH112": 3, "PHY117": 1, "CHM113": 3, "CHM114": 1, "BIO112": 2,
    "GSS111": 1, "GSS112": 2, "GSS113": 1, "UGC111": 1, "CSC112": 2,
    "GSS116": 1, "PHY112": 2, "MTH122": 3, "PHY122": 2, "PHY127": 1,
    "CHM121": 3, "CHM124": 1, "GSS121": 2, "GSS126": 1, "UGC121": 2,
    "CSC122": 2, "CSC123": 2, "MTH123": 3, "MTH211": 3, "STA212": 2,
    "PHY212": 2, "GSS212": 2, "GSS217": 3, "CSC211": 2, "CSC212": 2,
    "CSC213": 2, "CSC218": 3, "MTH221": 3, "MTH222": 3, "STA224": 3,
    "PHY222": 2, "GNT221": 3, "CSC221": 3, "CSC222": 2, "CSC226": 2,
    "CSC311": 3, "CSC312": 2, "CSC313": 3, "CSC314": 3, "CSC316": 3,
    "CSC317": 3, "CSC318": 3, "GNT311": 2
}

# Load cumulative data from Excel for a specific student
def load_student_data_from_excel(file_path, regnum):
    # Load the data without headers
    excel_data = pd.read_excel(file_path, sheet_name='F300L', header=None)
    
    # Define cell ranges based on your specifications
    student_name = excel_data.loc[6:328, 1]  # C7:C329
    reg_numbers = excel_data.loc[6:328, 2]  # C7:C329
    ctcl_values = excel_data.loc[6:328, 29]  # AD7:AD329
    cgp_values = excel_data.loc[6:328, 30]   # AE7:AE329
    cgpa_values = excel_data.loc[6:328, 31]  # AF7:AF329
    remarks = excel_data.loc[6:328, 32]      # AG7:AG329

    # Combine these columns into a DataFrame
    student_df = pd.DataFrame({
        'NAME':student_name,
        'REG. NO.': reg_numbers,
        'CTCL': ctcl_values,
        'CGP': cgp_values,
        'CGPA': cgpa_values,
        'Remark': remarks
    })

    # Filter for the student with the given registration number
    student_data = student_df[student_df['REG. NO.'] == regnum]

    # Check if the registration number was found
    if student_data.empty:
        return None  # or raise an error, depending on how you want to handle it

    return student_data.iloc[0]  # Return the student data as a single row


# Define fuzzy variables and membership functions
def define_fuzzy_variables():
    # CGPA
    cgpa = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'CGPA')
    course_duration = ctrl.Antecedent(np.arange(0, 11, 1), 'Course Duration')
    credit_load = ctrl.Antecedent(np.arange(0, 31, 1), 'Credit Load')
    spill_over = ctrl.Antecedent(np.arange(0, 11, 1), 'Spill Over')
    advice = ctrl.Consequent(np.arange(0, 101, 1), 'Advice')

    # Membership functions
    cgpa['Low'] = fuzz.trimf(cgpa.universe, [0, 0, 2])
    cgpa['Medium'] = fuzz.trimf(cgpa.universe, [1.5, 3, 3.5])
    cgpa['High'] = fuzz.trimf(cgpa.universe, [3, 5, 5])

    course_duration['Short'] = fuzz.trimf(course_duration.universe, [0, 0, 3])
    course_duration['Normal'] = fuzz.trimf(course_duration.universe, [2, 5, 5])
    course_duration['Long'] = fuzz.trimf(course_duration.universe, [4, 10, 10])

    credit_load['Low'] = fuzz.trimf(credit_load.universe, [0, 0, 15])
    credit_load['Medium'] = fuzz.trimf(credit_load.universe, [10, 15, 20])
    credit_load['High'] = fuzz.trimf(credit_load.universe, [15, 30, 30])

    spill_over['Few'] = fuzz.trimf(spill_over.universe, [0, 0, 5])
    spill_over['Many'] = fuzz.trimf(spill_over.universe, [5, 10, 10])

    advice['Poor'] = fuzz.trimf(advice.universe, [0, 0, 25])
    advice['Average'] = fuzz.trimf(advice.universe, [20, 50, 80])
    advice['Excellent'] = fuzz.trimf(advice.universe, [75, 100, 100])
    
    return (cgpa, course_duration, credit_load, spill_over, advice)

# Define fuzzy rules
def define_fuzzy_rules(cgpa, course_duration, credit_load, spill_over, advice):
    rule1 = ctrl.Rule(cgpa['Low'] & course_duration['Long'], advice['Poor'])
    rule2 = ctrl.Rule(cgpa['Medium'] & course_duration['Normal'], advice['Average'])
    rule3 = ctrl.Rule(cgpa['High'] & credit_load['High'], advice['Excellent'])
    rule4 = ctrl.Rule(spill_over['Many'], advice['Poor'])
    rule5 = ctrl.Rule(spill_over['Few'], advice['Excellent'])
    
    return [rule1, rule2, rule3, rule4, rule5]


# Generate advice using fuzzy logic and expert rules
# Generate advice using fuzzy logic and expert rules
def get_combined_advice(query_type, value, cgpa_value, knowledge_base, remarks,cumulative_tcl, student_name):
    value_str = str(int(value))
    cgpa_advice = []
    query_advice = []

    # Define fuzzy variables and rules
    (cgpa, course_duration, credit_load, spill_over, advice) = define_fuzzy_variables()
    rules = define_fuzzy_rules(cgpa, course_duration, credit_load, spill_over, advice)
    
    advice_ctrl = ctrl.ControlSystem(rules)
    advice_simulation = ctrl.ControlSystemSimulation(advice_ctrl)

    # Process CGPA-based advice
    for entry in knowledge_base:
        if entry['Type'].strip() == 'CGPA':
            range_value = entry.get('Range', '').strip()
            try:
                range_min, range_max = map(float, range_value.split('-'))
            except ValueError:
                continue
            if range_min <= cgpa_value <= range_max:
                cgpa_advice.append(entry.get('Advice', "No advice available.").strip())
                
        # Process CGPA-based advice
    possible_additional_units = 0
    if cgpa_value >= 4.0:
        possible_additional_units = 3
    elif 3.0 <= cgpa_value < 4.0:
        possible_additional_units = 2
    elif 2.0 <= cgpa_value < 3.0:
        possible_additional_units = 1

    allowed_credit_load = 24 + possible_additional_units  # Calculate the total allowed units for the semester
    
    if possible_additional_units==0:
        cgpa_advice.append(f"\nYou do not qualify for additional units this semester, work harder to build your CGPA.")
    else:
        cgpa_advice.append(f"\nYou qualify for {possible_additional_units} additional units this semester, bringing the maximum allowable credit load to {allowed_credit_load} units.")

    # Process remarks to calculate current total credit units
    total_credit_units_remarks = 0
    remark_codes = remarks.split(',')
    
    if remarks and (remarks != "COMMENDATION" and remarks != "PASS"):
        remark_codes = remarks.split(',')
        formatted_remarks = ", ".join([code.strip() for code in remark_codes])
        query_advice.append(f"\nYour outstanding courses are: {formatted_remarks}.\n")
    elif remarks == "COMMENDATION":
        query_advice.append(f"\nCongratulations!!! You are being recommended by the University Senate, there is always a letter to this effect.\n")
    else:
        query_advice.append(f"\nAs of my cut-off date, you donot have any outstanding. Continue to uput more effort!\n")
    


        
    for code in remark_codes:
        code = code.strip()  # Clean whitespace
        if code in course_credit_units:
            total_credit_units_remarks += course_credit_units[code]
            
    
    thisSemesterLoad = 0
   # Calculate thisSemesterLoad based on cumulative_tcl
    if 20 <= cumulative_tcl <= 20:
        thisSemesterLoad = 20
    elif 21 <= cumulative_tcl <= 41:
        thisSemesterLoad = 21
    elif 62 <= cumulative_tcl <= 65:
        thisSemesterLoad = 21
    elif 83 <= cumulative_tcl <= 89:
        thisSemesterLoad = 22
    elif 105 <= cumulative_tcl <= 114:
        thisSemesterLoad = 22
    else:
        # Handle cases outside the given ranges
        thisSemesterLoad = 0  # or some default value

   
    total_credit_units_thisSemester = total_credit_units_remarks + thisSemesterLoad
    
    # Check if the student exceeds the allowable credit load and suggest dropping courses
    if total_credit_units_thisSemester > allowed_credit_load:
        excess_units = total_credit_units_thisSemester - allowed_credit_load
        query_advice.append(f"Your credit load this semester (carry over inclusive)  is {total_credit_units_thisSemester} which exceeds the allowable {allowed_credit_load} units. Please consider dropping {excess_units} credit units from current courses to comply with the limit.\n\n")
    else:
        query_advice.append(f"\nYour current credit load is within the allowable limit of {allowed_credit_load} units.\n")

                

    # Process query-based advice based on `query_type`
    for entry in knowledge_base:
        if entry['Type'].strip() == query_type:
            range_value = entry.get('Range', '').strip()
            if query_type == 'Course Duration' and range_value == f"{value_str} years":
                total_years_str = entry['Total_Allowed_Years'].strip().split()[0]
                total_years = int(total_years_str)
                if int(value) > total_years:
                    query_advice.append(entry.get('Advice', "No advice available.").strip())
                else:
                    query_advice.append(entry.get('Advice', "No advice available.").strip())
                    query_advice.append(f"You are within the allowed {total_years} years. Keep progressing!")
            elif query_type == 'Credit Load':
                if range_value == '25 and above' and value >= 25:
                    query_advice.append(entry.get('Advice', "No advice available.").strip())
                elif range_value == '15 and 24' and 15 <= value <= 24:
                    query_advice.append(entry.get('Advice', "No advice available.").strip())
                elif range_value == 'Below 15' and value < 15:
                    query_advice.append(entry.get('Advice', "No advice available.").strip())
            elif query_type == 'Spill-over':
                if 'Less than 6 credit units' in range_value and float(value) < 6:
                    query_advice.append(entry.get('Advice', "No advice available.").strip())
                elif '6 and above credit units' in range_value and float(value) >= 6:
                    query_advice.append(entry.get('Advice', "No advice available.").strip())



     
    # Prepare combined advice with CGPA included
    combined_advice = "\n".join(cgpa_advice + query_advice).strip()
    final_advice = (
         f" Dear {student_name}, Your CGPA is {cgpa_value:.2f}." + combined_advice
        if combined_advice
        else "No advice available for this query."
    )

    return final_advice



from flask import Flask, request, redirect, render_template



@app.route('/submit-rating', methods=['POST'])
def submit_rating(): 
    rating = request.form['rating'] 
    regnum = request.form['regnum'] 
    with open('ratings.txt', 'a') as file: 
        file.write(f'Registration Number: {regnum}, Rating: {rating}\n') 
    return redirect('/')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_advice', methods=['POST'])
def get_advice_route():
    data = request.json
    regnum = data['regnum']
    query_type = data['query_type']
    value = data['value']

    try:
        value = float(value)
    except ValueError:
        return {"advice": "Invalid value. Please enter a valid number."}

    # Load CSV knowledge base
    knowledge_base = load_knowledge_base('KB/knowledge_base.csv')

    # Load Excel data for the specific student
    student_data = load_student_data_from_excel('DB/student_data.xlsm', regnum)

    if student_data is None:
        return {"advice": "No student found with this registration number."}

    # Extract cumulative values from the student's data
    student_name = student_data['NAME']
    cumulative_tcl = student_data['CTCL']
    cumulative_gp = student_data['CGP']
    cumulative_gpa = student_data['CGPA']
    remarks = student_data['Remark']  # Get the remarks

    # Generate advice based on knowledge base and cumulative data from Excel
    advice = get_combined_advice(query_type, value, cumulative_gpa, knowledge_base, remarks, cumulative_tcl,student_name)


    return {
        "advice": advice,
        "cumulative_tcl": cumulative_tcl,
        "cumulative_gp": cumulative_gp,
        "cumulative_gpa": cumulative_gpa
    }


if __name__ == '__main__':
    app.run(debug=True)
