from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

# Load model and tokenizer (adjust based on your system's capacity)
# Load the tokenizer and model
model_name = "numind/NuExtract-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
# Create a generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Your JSON template
json_template = {
  "root": {
    "Complaints": [
      {
        "complaint": "Description of the patient's primary complaint",
        "complaint_duration": "Duration of the complaint (numeric), e.g., 2",
        "complaint_duration_type": "Duration unit, e.g., 'D' for days, 'W' for weeks, 'M' for months, 'Y' for years"
      },
      {
        "complaint": "Additional complaint description",
        "complaint_duration": "Optional duration of this complaint",
        "complaint_duration_type": "Optional duration type",
        "code": "Complaint code if available (e.g., ICD-10)"
      }
    ],
    "vital": {
      "weight": "Patient's weight in kilograms",
      "height": "Patient's height in centimeters",
      "bp": "Blood pressure reading as a string (e.g., '120/80')",
      "pulse": "Pulse rate (beats per minute)",
      "respiratory_rate": "Respiratory rate (breaths per minute)",
      "spo2": "Oxygen saturation percentage"
    },
    "examination": {
      "examination_detail": "Detailed notes from the physical examination"
    },
    "prov_diagnosis": {
      "provisional_diagnosis": "Initial/provisional diagnosis based on current findings"
    },
    "diagnosis": [
      " List of confirmed diagnoses if available"
    ],
    "investigations": [
      {
        "investigation": "Recommended investigation/test (e.g., 'CBC', 'Lipid profile')"
      }
    ],
    "MedicalAdvice": [
      {
        "medication_advice": "Name of the prescribed medicine",
        "intake": "Dosage or intake instruction (e.g., '500 mg', 'once daily')",
        "duration": "Duration for which the medication should be taken (numeric)",
        "duration_type": "Duration unit: 'D', 'W', 'M', 'Y'",
        "route": "Route of administration (e.g., 'Oral', 'IM', 'IV')",
        "remark": "Additional instructions or notes",
        "medicine_code": "Unique medicine code if applicable"
      }
    ],
    "medical_history": [
      {
        "medical_history": "Medical condition (e.g., 'Diabetes', 'Hypertension')",
        "medical_history_duration": "How long the patient has had the condition (numeric)",
        "medical_history_type": "Duration unit: 'D', 'W', 'M', 'Y'",
        "medical_history_free_text": "Additional details about the condition",
        "code": "Code for the condition if available (e.g., ICD code)"
      }
    ],
    "Complaint": {
      "complaint_free_text": "Free text describing patient's general complaints not coded above"
    },
    "investigation": {
      "investigation_free_text": "Free text describing suggested investigations not coded above"
    },
    "medicine_code_free_text": {
      "medicine_code_free_text": "Free text for any medicine not in the system or not coded"
    }
  }
}

# Paragraph to extract from
paragraph = """
Subjective:

Complaints: Patient reports needing to lose 10% body weight. Experiences knee joint pain, potentially due to B12 deficiency.
Medical History: Diabetes for 22 years. HbA1c level was 8.7 six months ago.
Allergies: No information provided.
Objective:

Vitals:
Weight: 71 kg
Height: 160 cm
BMI: 28
Blood Pressure: One time by 70 (Value incomplete)
Temperature: Not provided
SpO2: Not provided
Clinical Examination:
General Physical Examination: Mention of required weight loss and B12 deficiency.
Cardiovascular: No information provided.
Respiratory: No information provided.
Central Nervous: Neuropathy screening discussed, potentially related to Diabetes.
Abdomen: No information provided.
Musculoskeletal: Knee joint pain assessment.
Assessment:

Investigations discussed:
Blood pressure monitoring
Lipid profile
HbA1c
Neuropathy screening
Plan:

Provisional/Final Diagnosis: B12 deficiency, uncontrolled diabetes, and overweight.
Advice:
Weight reduction by at least 6-7 kg.
Neuropathy screening for knee joint pain.
Follow-up for vitamin shots, especially B12.
Medications (Rx):

Medication Name	Type	Dosage	Duration	Route	Remarks
Gemer-P	Tablet	Prescribed dosage not provided	Not specified	Oral	Changed from previous
Presoval ten	Tablet	Prescribed dosage not provided	Not specified	Oral	Updated medication
Oxra met S XR	Tablet	500 mg	Not specified	Oral	For diabetes
Vitamin B12	Injection	As advised by the doctor	1 now, follow up mentioned	IM	For B12 deficiency
Vitamin D	Injection	Weekly for 8 weeks	8 weeks	IM	For deficiency
"""

# Prompt for model
prompt = f"""
Extract the relevant information from the following paragraph and return only a JSON object in this format:

{json.dumps(json_template, indent=2)}

Paragraph:
\"\"\"{paragraph}\"\"\"

Respond ONLY with a one valid JSON object.
"""

# Generate response
output = generator(prompt, max_new_tokens=300, do_sample=False)[0]['generated_text']

# Extract JSON from response
try:
    start = output.find('{')
    end = output.rfind('}') + 1
    json_str = output[start:end]
    extracted_json = json.loads(json_str)
    print("✅ Extracted JSON:\n", json.dumps(extracted_json, indent=2))
except Exception as e:
    print("❌ Failed to extract JSON.")
    print("Raw Output:\n", output)
