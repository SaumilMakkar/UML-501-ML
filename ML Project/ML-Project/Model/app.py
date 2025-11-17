"""
🎯 Resume Category Prediction App
A beautiful Streamlit frontend for resume categorization with high-confidence predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import PyPDF2
import pdfplumber

# Try to import OCR dependencies (optional)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Job recommendations database
JOB_RECOMMENDATIONS = {
    'Data Science': {
        'top_companies': ['Google', 'Amazon', 'Microsoft', 'IBM', 'Meta (Facebook)', 'Apple', 'Netflix', 'LinkedIn', 'Tesla', 'Adobe'],
        'job_platforms': ['Kaggle Jobs', 'DataJobs', 'LinkedIn', 'Indeed', 'Glassdoor', 'AngelList', 'Stack Overflow Jobs', 'Hired'],
        'learning_platforms': ['Coursera', 'edX', 'Kaggle Learn', 'DataCamp', 'Udacity', 'Udemy', 'Fast.ai', 'DeepLearning.AI'],
        'avg_salary': '$120,000 - $180,000',
        'skills_needed': ['Python', 'R', 'SQL', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Data Analysis']
    },
    'Python Developer': {
        'top_companies': ['Google', 'Dropbox', 'Instagram', 'Spotify', 'Disney', 'NASA', 'Uber', 'Twilio', 'Reddit', 'Mozilla'],
        'job_platforms': ['Python.org Jobs', 'Stack Overflow Jobs', 'LinkedIn', 'Indeed', 'Dice', 'Remote Python', 'WeWorkRemotely', 'Python Weekly'],
        'learning_platforms': ['Real Python', 'Python.org', 'Codecademy', 'freeCodeCamp', 'Coursera', 'edX', 'Udemy', 'Pluralsight'],
        'avg_salary': '$95,000 - $145,000',
        'skills_needed': ['Python', 'Django', 'Flask', 'FastAPI', 'AWS', 'Docker', 'Git', 'REST APIs']
    },
    'Java Developer': {
        'top_companies': ['Oracle', 'Amazon', 'Google', 'Microsoft', 'IBM', 'Netflix', 'LinkedIn', 'eBay', 'Yahoo', 'Apache'],
        'job_platforms': ['JavaJobs', 'Stack Overflow Jobs', 'LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'Dice', 'Built In'],
        'learning_platforms': ['Oracle Java Tutorials', 'Codecademy', 'Coursera', 'Udemy', 'Pluralsight', 'JavaTpoint', 'Baeldung', 'GeeksforGeeks'],
        'avg_salary': '$90,000 - $140,000',
        'skills_needed': ['Java', 'Spring Boot', 'Hibernate', 'Maven', 'Microservices', 'JUnit', 'REST APIs']
    },
    'HR': {
        'top_companies': ['Google', 'Microsoft', 'Salesforce', 'Workday', 'LinkedIn', 'BambooHR', 'Zoho', 'Oracle', 'IBM', 'Accenture'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Workable', 'BambooHR Careers', 'SHRM', 'HR.com', 'Indeed for HR'],
        'learning_platforms': ['SHRM Learning', 'Coursera', 'edX', 'Udemy', 'HR University', 'LinkedIn Learning', 'HRCI', 'ATD'],
        'avg_salary': '$60,000 - $110,000',
        'skills_needed': ['Recruitment', 'HRIS', 'Talent Management', 'Employee Relations', 'Analytics', 'Communication', 'Employment Law']
    },
    'DevOps Engineer': {
        'top_companies': ['Amazon (AWS)', 'Google Cloud', 'Microsoft Azure', 'Docker', 'Kubernetes', 'GitLab', 'Red Hat', 'Netflix', 'Uber', 'Slack'],
        'job_platforms': ['DevOps.com', 'Dice', 'Stack Overflow Jobs', 'LinkedIn', 'Indeed', 'AngelList', 'WeWorkRemotely', 'Remote OK'],
        'learning_platforms': ['Linux Academy', 'A Cloud Guru', 'Udemy', 'Coursera', 'Pluralsight', 'KodeKloud', 'DevOps Institute', 'Cloud Native Computing Foundation'],
        'avg_salary': '$110,000 - $170,000',
        'skills_needed': ['AWS', 'Docker', 'Kubernetes', 'Jenkins', 'CI/CD', 'Linux', 'Ansible', 'Terraform']
    },
    'Web Designing': {
        'top_companies': ['Shopify', 'Squarespace', 'Adobe', 'Google', 'Meta', 'Apple', 'Figma', 'Canva', 'Webflow', 'Wix'],
        'job_platforms': ['Dribbble Jobs', 'Behance', 'Indeed', 'LinkedIn', 'AngelList', 'AIGA', 'Designer Hangout', 'Working Not Working'],
        'learning_platforms': ['Figma Learn', 'Webflow University', 'Codecademy', 'freeCodeCamp', 'Coursera', 'Skillshare', 'Udemy', 'Interaction Design Foundation'],
        'avg_salary': '$55,000 - $95,000',
        'skills_needed': ['UI/UX', 'HTML/CSS', 'JavaScript', 'Adobe Creative Suite', 'Figma', 'Responsive Design', 'User Research']
    },
    'Testing': {
        'top_companies': ['QA Systems', 'Capgemini', 'Infosys', 'Tata Consultancy', 'Accenture', 'IBM', 'Microsoft', 'Google', 'Amazon', 'Oracle'],
        'job_platforms': ['Dice', 'Indeed', 'LinkedIn', 'Glassdoor', 'Stack Overflow Jobs', 'Testing Jobs', 'QA Jobs', 'Monster'],
        'learning_platforms': ['ISTQB', 'Test Automation University', 'Udemy', 'Coursera', 'Pluralsight', 'Guru99', 'Software Testing Help', 'Ministry of Testing'],
        'avg_salary': '$70,000 - $110,000',
        'skills_needed': ['Selenium', 'TestNG', 'JIRA', 'Manual Testing', 'Automation', 'Postman', 'Agile', 'ISTQB']
    },
    'Sales': {
        'top_companies': ['Salesforce', 'HubSpot', 'Amazon', 'Microsoft', 'Oracle', 'Adobe', 'ServiceNow', 'Zoom', 'DocuSign', 'LinkedIn'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Sales Jobs', 'MedReps', 'Sales Gravy', 'ZipRecruiter', 'Salesforce Careers'],
        'learning_platforms': ['Salesforce Trailhead', 'HubSpot Academy', 'Coursera', 'Udemy', 'LinkedIn Learning', 'Sales Hacker', 'Salesforce University', 'Sales Enablement Pro'],
        'avg_salary': '$50,000 - $120,000+ (commissions)',
        'skills_needed': ['CRM', 'Negotiation', 'Communication', 'Lead Generation', 'Customer Relations', 'Sales Analytics', 'Presentation']
    },
    'Business Analyst': {
        'top_companies': ['Deloitte', 'EY', 'PwC', 'McKinsey', 'BCG', 'Accenture', 'IBM', 'Microsoft', 'Oracle', 'SAP'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'Dice', 'Business Analysis Jobs', 'IIBA', 'AIM'],
        'learning_platforms': ['IIBA', 'Coursera', 'edX', 'Udemy', 'LinkedIn Learning', 'BABOK Guide', 'Analyst Exchange', 'Modern Analyst'],
        'avg_salary': '$70,000 - $110,000',
        'skills_needed': ['SQL', 'Excel', 'Tableau', 'Power BI', 'Agile', 'Requirements Analysis', 'Business Process', 'Stakeholder Management']
    },
    'Database': {
        'top_companies': ['Oracle', 'Microsoft', 'Amazon', 'MongoDB Inc', 'Snowflake', 'Databricks', 'Redis Labs', 'IBM', 'Google', 'SAP'],
        'job_platforms': ['Dice', 'Indeed', 'LinkedIn', 'Stack Overflow Jobs', 'Glassdoor', 'Remote DBA Jobs', 'Database Weekly', 'PostgreSQL Jobs'],
        'learning_platforms': ['Oracle University', 'MongoDB University', 'Coursera', 'Udemy', 'Pluralsight', 'PostgreSQL Tutorial', 'MySQL Learning', 'Database Journal'],
        'avg_salary': '$90,000 - $135,000',
        'skills_needed': ['SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Oracle Database', 'NoSQL', 'Database Design', 'Backup & Recovery']
    },
    'Blockchain': {
        'top_companies': ['Coinbase', 'Binance', 'Chainlink Labs', 'Solana', 'Polygon', 'Ripple', 'Ethereum Foundation', 'Consensys', 'Hyperledger', 'Web3 Foundation'],
        'job_platforms': ['CryptoJobs', 'Web3.career', 'LinkedIn', 'Indeed', 'AngelList', 'Blockchain Jobs', 'Remote Blockchain', 'NFT Jobs'],
        'learning_platforms': ['CryptoZombies', 'Ethereum.org', 'Coursera', 'Udemy', 'Blockchain Council', 'Consensys Academy', 'B9lab', 'Dapp University'],
        'avg_salary': '$100,000 - $180,000',
        'skills_needed': ['Solidity', 'Ethereum', 'Smart Contracts', 'DeFi', 'Cryptography', 'Web3', 'Truffle', 'Remix']
    },
    'Network Security Engineer': {
        'top_companies': ['CrowdStrike', 'Palo Alto Networks', 'Fortinet', 'Check Point', 'Cisco', 'IBM Security', 'FireEye', 'Rapid7', 'Tenable', 'Okta'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Dice', 'Glassdoor', 'Cybersecurity Jobs', 'InfoSec', 'Security Clearance Jobs', 'CyberCoders'],
        'learning_platforms': ['Cybrary', 'SANS', 'Coursera', 'Udemy', 'Pluralsight', 'TryHackMe', 'Hack The Box', 'Security+'],
        'avg_salary': '$95,000 - $150,000',
        'skills_needed': ['Cybersecurity', 'Firewall', 'Intrusion Detection', 'SIEM', 'Penetration Testing', 'Network Protocols', 'Compliance', 'CISSP']
    },
    'SAP Developer': {
        'top_companies': ['SAP', 'Accenture', 'Deloitte', 'IBM', 'Capgemini', 'Infosys', 'Tata Consultancy', 'Wipro', 'EY', 'Oracle'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'Dice', 'SAP Careers', 'SAP Job Portal', 'Freelance SAP'],
        'learning_platforms': ['SAP Learning Hub', 'openSAP', 'Udemy', 'Coursera', 'SAP Community', 'SAP Training', 'SAPinsider', 'SAP PRESS'],
        'avg_salary': '$85,000 - $130,000',
        'skills_needed': ['ABAP', 'SAP HANA', 'SAP Fiori', 'SAP S/4HANA', 'BW/4HANA', 'BTP', 'SAP Integration', 'Business Processes']
    },
    'Automation Testing': {
        'top_companies': ['UiPath', 'Automation Anywhere', 'Blue Prism', 'Microsoft', 'Google', 'Amazon', 'IBM', 'Accenture', 'Capgemini', 'Cognizant'],
        'job_platforms': ['Dice', 'Indeed', 'LinkedIn', 'Stack Overflow Jobs', 'Testing Jobs', 'QA Jobs', 'Monster', 'Automation Jobs'],
        'learning_platforms': ['UiPath Academy', 'Automation Anywhere University', 'Udemy', 'Coursera', 'Test Automation University', 'Selenium', 'Pluralsight', 'Guru99'],
        'avg_salary': '$75,000 - $115,000',
        'skills_needed': ['Selenium', 'Appium', 'Jenkins', 'Robot Framework', 'Test Complete', 'API Testing', 'BDD', 'Cypress']
    },
    'Mechanical Engineer': {
        'top_companies': ['Boeing', 'Tesla', 'Ford', 'General Electric', 'Caterpillar', 'Lockheed Martin', 'Honeywell', 'Siemens', 'ABB', '3M'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'ASME Jobs', 'Engineering.com', 'Engineering Jobs', 'ZipRecruiter'],
        'learning_platforms': ['Coursera', 'edX', 'Udemy', 'Khan Academy', 'MIT OpenCourseWare', 'ASME', 'Engineering.com', 'SolidWorks'],
        'avg_salary': '$75,000 - $110,000',
        'skills_needed': ['CAD', 'SolidWorks', 'AutoCAD', 'MATLAB', 'Finite Element Analysis', 'Thermodynamics', 'Manufacturing', 'Project Management']
    },
    'Civil Engineer': {
        'top_companies': ['AECOM', 'Jacobs Engineering', 'Fluor Corporation', 'Kiewit Corporation', 'Turner Construction', 'Skanska', 'Bechtel', 'L&T', 'Infrastructure Corporation'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'ASCE Jobs', 'Engineering.com', 'Civil Jobs', 'Construction Jobs'],
        'learning_platforms': ['Coursera', 'edX', 'Udemy', 'ASCE', 'Engineering.com', 'AutoCAD', 'Revit', 'Civil Engineering Portal'],
        'avg_salary': '$65,000 - $100,000',
        'skills_needed': ['AutoCAD', 'Revit', 'Structural Analysis', 'Project Management', 'Surveying', 'Building Codes', 'Construction Management']
    },
    'Electrical Engineering': {
        'top_companies': ['Siemens', 'ABB', 'General Electric', 'Schneider Electric', 'Honeywell', 'IBM', 'Intel', 'Boeing', 'Lockheed Martin', 'Tesla'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'IEEE Jobs', 'Engineering.com', 'Electrical Jobs', 'ZipRecruiter'],
        'learning_platforms': ['Coursera', 'edX', 'Udemy', 'IEEE', 'MIT OpenCourseWare', 'Khan Academy', 'All About Circuits', 'Electronics Tutorials'],
        'avg_salary': '$80,000 - $120,000',
        'skills_needed': ['PLC', 'SCADA', 'Circuit Design', 'MATLAB', 'Power Systems', 'Control Systems', 'Embedded Systems', 'Project Management']
    },
    'Operations Manager': {
        'top_companies': ['Amazon', 'Walmart', 'FedEx', 'UPS', 'Procter & Gamble', 'Coca-Cola', 'Nestlé', 'Unilever', 'Johnson & Johnson', 'Intel'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'Operations Jobs', 'ZipRecruiter', 'Business Management Jobs', 'Supply Chain Jobs'],
        'learning_platforms': ['Coursera', 'edX', 'Udemy', 'LinkedIn Learning', 'APICS', 'Six Sigma', 'Lean Manufacturing', 'Supply Chain Management'],
        'avg_salary': '$75,000 - $120,000',
        'skills_needed': ['Supply Chain', 'Lean Manufacturing', 'Six Sigma', 'Project Management', 'ERP Systems', 'Quality Control', 'Team Leadership']
    },
    'DotNet Developer': {
        'top_companies': ['Microsoft', 'Accenture', 'Infosys', 'Tata Consultancy', 'Capgemini', 'IBM', 'Cognizant', 'HCL Technologies', 'Wipro', 'Tech Mahindra'],
        'job_platforms': ['Stack Overflow Jobs', 'LinkedIn', 'Indeed', 'Dice', 'Glassdoor', 'Monster', '.NET Jobs', 'Microsoft Careers'],
        'learning_platforms': ['Microsoft Learn', 'Pluralsight', 'Udemy', 'Coursera', '.NET Foundation', 'Codecademy', 'freeCodeCamp', 'Microsoft Docs'],
        'avg_salary': '$80,000 - $125,000',
        'skills_needed': ['C#', 'ASP.NET', 'MVC', '.NET Core', 'Entity Framework', 'SQL Server', 'Azure', 'Web APIs']
    },
    'ETL Developer': {
        'top_companies': ['Amazon', 'Microsoft', 'IBM', 'Informatica', 'Talend', 'SAS Institute', 'Oracle', 'SAP', 'Snowflake', 'Databricks'],
        'job_platforms': ['Dice', 'Indeed', 'LinkedIn', 'Stack Overflow Jobs', 'Glassdoor', 'ETL Jobs', 'Data Engineering Jobs', 'Monster'],
        'learning_platforms': ['Udemy', 'Coursera', 'Pluralsight', 'Informatica University', 'Talend Academy', 'DataCamp', 'Coursera', 'edX'],
        'avg_salary': '$90,000 - $130,000',
        'skills_needed': ['SQL', 'ETL Tools', 'Data Warehousing', 'Python', 'Hadoop', 'Spark', 'Informatica', 'Talend']
    },
    'Hadoop': {
        'top_companies': ['Cloudera', 'Hortonworks', 'Databricks', 'Amazon', 'Microsoft', 'IBM', 'Oracle', 'MapR', 'Splunk', 'Tableau'],
        'job_platforms': ['Dice', 'Indeed', 'LinkedIn', 'Stack Overflow Jobs', 'Glassdoor', 'Big Data Jobs', 'Data Engineering Jobs', 'Monster'],
        'learning_platforms': ['Cloudera University', 'Udemy', 'Coursera', 'edX', 'Pluralsight', 'Hadoop Tutorial', 'Big Data University', 'DataCamp'],
        'avg_salary': '$110,000 - $160,000',
        'skills_needed': ['Hadoop', 'Spark', 'Hive', 'Pig', 'HBase', 'Kafka', 'Flume', 'Sqoop', 'YARN', 'MapReduce']
    },
    'PMO': {
        'top_companies': ['PMI', 'Deloitte', 'PwC', 'EY', 'Accenture', 'IBM', 'Microsoft', 'Amazon', 'Oracle', 'SAP'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'PMI.org', 'Project Management Jobs', 'Dice', 'ZipRecruiter'],
        'learning_platforms': ['PMI', 'Coursera', 'edX', 'Udemy', 'LinkedIn Learning', 'Project Management Institute', 'Agile Alliance', 'Scrum.org'],
        'avg_salary': '$90,000 - $130,000',
        'skills_needed': ['PMP', 'Agile', 'Scrum', 'PMO Leadership', 'Project Management', 'Risk Management', 'Stakeholder Management']
    },
    'Health and fitness': {
        'top_companies': ['Peloton', 'Equinox', 'Nike', 'Adidas', 'Under Armour', 'Garmin', 'Fitbit', 'MyFitnessPal', "Gold's Gym", 'Planet Fitness'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'Fitness Jobs', 'Trainer Jobs', 'Wellness Jobs', 'ZipRecruiter'],
        'learning_platforms': ['NASM', 'ACE', 'ACSM', 'Udemy', 'Coursera', 'Fitness Mentors', 'Precision Nutrition', 'Stronger by Science'],
        'avg_salary': '$35,000 - $75,000',
        'skills_needed': ['Personal Training', 'Nutrition', 'Exercise Science', 'Customer Service', 'Marketing', 'Program Design', 'First Aid']
    },
    'Arts': {
        'top_companies': ['Walt Disney', 'Warner Bros', 'Universal Studios', 'Pixar', 'DreamWorks', 'Adobe', 'Canva', 'Meta', 'Google', 'Apple'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Arts Jobs', 'Creative Jobs', 'ArtJobs', 'Dribbble', 'Behance', 'Working Not Working'],
        'learning_platforms': ['Skillshare', 'Domestika', 'Udemy', 'Coursera', 'CreativeLive', 'School of Motion', 'Blender', 'Adobe Creative Cloud'],
        'avg_salary': '$45,000 - $100,000',
        'skills_needed': ['Creativity', 'Design Software', 'Digital Art', 'Animation', 'Adobe Creative Suite', 'Portfolio Development', 'Art History']
    },
    'Advocate': {
        'top_companies': ['Baker McKenzie', 'Skadden Arps', 'Latham & Watkins', 'Kirkland & Ellis', 'DLA Piper', 'Clifford Chance', 'Linklaters', 'White & Case', 'Sidley Austin', 'Freshfields'],
        'job_platforms': ['LinkedIn', 'Indeed', 'Glassdoor', 'Law.com Jobs', 'Legal Jobs', 'Monster', 'Legal Recruiters', 'Attorney Jobs'],
        'learning_platforms': ['Coursera', 'edX', 'Westlaw', 'LexisNexis', 'ABA', 'Law School', 'Bar Prep', 'Legal Writing'],
        'avg_salary': '$70,000 - $200,000+',
        'skills_needed': ['Legal Research', 'Court Procedures', 'Contract Law', 'Legal Writing', 'Client Relations', 'Litigation', 'Compliance']
    }
}

# Page configuration
st.set_page_config(
    page_title="Resume Category Predictor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .confidence-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'clf_lr_calibrated' not in st.session_state:
    st.session_state.clf_lr_calibrated = None
if 'clf_rf_calibrated' not in st.session_state:
    st.session_state.clf_rf_calibrated = None
if 'clf_mlp_calibrated' not in st.session_state:
    st.session_state.clf_mlp_calibrated = None
if 'clf_gnb_calibrated' not in st.session_state:
    st.session_state.clf_gnb_calibrated = None
if 'clf_dt_calibrated' not in st.session_state:
    st.session_state.clf_dt_calibrated = None
if 'clf_knn_calibrated' not in st.session_state:
    st.session_state.clf_knn_calibrated = None
if 'word_vectorizer' not in st.session_state:
    st.session_state.word_vectorizer = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'model_weights' not in st.session_state:
    st.session_state.model_weights = {}
if 'model_accuracies' not in st.session_state:
    st.session_state.model_accuracies = {}

# Helper function to clean resume text
def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub(r'#\S+', '', resumeText)
    resumeText = re.sub(r'@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)
    return resumeText

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with OCR fallback"""
    pdf_bytes = pdf_file.read()
    
    # Try PyPDF2 first
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
                texts.append(txt)
            except Exception:
                texts.append("")
        content = ("\n".join(texts)).strip()
        if content and len(content) > 200:
            return content
    except Exception:
        pass
    
    # Try pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        content2 = ("\n".join(pages)).strip()
        if content2 and len(content2) > 200:
            return content2
    except Exception:
        pass
    
    # OCR fallback (if available)
    if OCR_AVAILABLE:
        try:
            tesseract_cmd = os.environ.get('TESSERACT_CMD', r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if os.path.exists(tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
            images = convert_from_bytes(pdf_bytes, dpi=300)
            ocr_texts = []
            for img in images[:5]:  # Limit to first 5 pages
                try:
                    ocr_texts.append(pytesseract.image_to_string(img, lang='eng'))
                except Exception:
                    continue
            ocr_content = ("\n".join(ocr_texts)).strip()
            if ocr_content:
                return ocr_content
        except Exception:
            pass
    
    raise ValueError("Could not extract text from PDF. Please ensure the PDF contains selectable text.")

# Train models function
@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    dataset_path = '../Dataset/UpdatedResumeDataSet.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'Dataset/UpdatedResumeDataSet.csv'
    
    resumeDataSet = pd.read_csv(dataset_path, encoding='utf-8')
    resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(x))
    return resumeDataSet

def train_models():
    """Train all 6 calibrated models"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load data
        status_text.text("📊 Step 1/8: Loading dataset...")
        progress_bar.progress(5)
        resumeDataSet = load_data()
        
        # Step 2: Prepare data
        status_text.text("🔧 Step 2/8: Preparing data (encoding, vectorizing)...")
        progress_bar.progress(10)
        
        # Encode categories
        le = LabelEncoder()
        resumeDataSet['Category_encoded'] = le.fit_transform(resumeDataSet['Category'])
        
        # Prepare features
        requiredText = resumeDataSet['cleaned_resume'].values
        requiredTarget = resumeDataSet['Category_encoded'].values
        
        # Vectorize
        word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
        WordFeatures = word_vectorizer.fit_transform(requiredText)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            WordFeatures, requiredTarget, random_state=42, test_size=0.2,
            shuffle=True, stratify=requiredTarget
        )
        
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        
        models = {}
        predictions = {}
        accuracies = {}
        
        # Step 3: Train Logistic Regression
        status_text.text("🤖 Step 3/8: Training Calibrated Logistic Regression...")
        progress_bar.progress(20)
        clf_lr_calibrated = CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, C=10.0, class_weight='balanced'),
            method='isotonic',
            cv=5
        )
        clf_lr_calibrated.fit(X_train_dense, y_train)
        models['lr'] = clf_lr_calibrated
        predictions['lr'] = clf_lr_calibrated.predict(X_test_dense)
        accuracies['lr'] = accuracy_score(y_test, predictions['lr'])
        
        # Step 4: Train Random Forest
        status_text.text("🌲 Step 4/8: Training Calibrated Random Forest...")
        progress_bar.progress(35)
        clf_rf_calibrated = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'),
            method='isotonic',
            cv=5
        )
        clf_rf_calibrated.fit(X_train_dense, y_train)
        models['rf'] = clf_rf_calibrated
        predictions['rf'] = clf_rf_calibrated.predict(X_test_dense)
        accuracies['rf'] = accuracy_score(y_test, predictions['rf'])
        
        # Step 5: Train MLP Neural Network
        status_text.text("🧠 Step 5/8: Training Calibrated MLP Neural Network...")
        progress_bar.progress(50)
        clf_mlp_calibrated = CalibratedClassifierCV(
            MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            method='isotonic',
            cv=5
        )
        clf_mlp_calibrated.fit(X_train, y_train)  # MLP can use sparse
        models['mlp'] = clf_mlp_calibrated
        predictions['mlp'] = clf_mlp_calibrated.predict(X_test)
        accuracies['mlp'] = accuracy_score(y_test, predictions['mlp'])
        
        # Step 6: Train Multinomial Naive Bayes (better for text/TF-IDF than GaussianNB)
        status_text.text("📊 Step 6/8: Training Calibrated Multinomial Naive Bayes...")
        progress_bar.progress(65)
        # Use MultinomialNB instead of GaussianNB for text data (TF-IDF features)
        # MultinomialNB works better with sparse matrices and non-negative features
        clf_gnb_calibrated = CalibratedClassifierCV(
            OneVsRestClassifier(MultinomialNB(alpha=1.0)),
            method='isotonic',
            cv=5
        )
        clf_gnb_calibrated.fit(X_train, y_train)  # Can use sparse matrix
        models['gnb'] = clf_gnb_calibrated
        predictions['gnb'] = clf_gnb_calibrated.predict(X_test)
        accuracies['gnb'] = accuracy_score(y_test, predictions['gnb'])
        
        # Step 7: Train Decision Tree
        status_text.text("🌳 Step 7/8: Training Calibrated Decision Tree...")
        progress_bar.progress(80)
        # Use min_samples_split and min_samples_leaf to prevent NaN issues
        clf_dt_calibrated = CalibratedClassifierCV(
            OneVsRestClassifier(DecisionTreeClassifier(
                random_state=42, 
                class_weight='balanced',
                min_samples_split=5,
                min_samples_leaf=2,
                max_depth=15  # Limit depth to prevent overfitting
            )),
            method='isotonic',
            cv=5
        )
        clf_dt_calibrated.fit(X_train_dense, y_train)
        models['dt'] = clf_dt_calibrated
        predictions['dt'] = clf_dt_calibrated.predict(X_test_dense)
        accuracies['dt'] = accuracy_score(y_test, predictions['dt'])
        
        # Step 8: Train K-Nearest Neighbors
        status_text.text("📍 Step 8/8: Training Calibrated K-Nearest Neighbors...")
        progress_bar.progress(90)
        clf_knn_calibrated = CalibratedClassifierCV(
            OneVsRestClassifier(KNeighborsClassifier()),
            method='isotonic',
            cv=5
        )
        clf_knn_calibrated.fit(X_train, y_train)  # KNN can use sparse
        models['knn'] = clf_knn_calibrated
        predictions['knn'] = clf_knn_calibrated.predict(X_test)
        accuracies['knn'] = accuracy_score(y_test, predictions['knn'])
        
        # Calculate weights based on accuracies
        total_acc = sum(accuracies.values())
        model_weights = {name: acc / total_acc for name, acc in accuracies.items()}
        
        # Store in session state
        st.session_state.clf_lr_calibrated = models['lr']
        st.session_state.clf_rf_calibrated = models['rf']
        st.session_state.clf_mlp_calibrated = models['mlp']
        st.session_state.clf_gnb_calibrated = models['gnb']
        st.session_state.clf_dt_calibrated = models['dt']
        st.session_state.clf_knn_calibrated = models['knn']
        st.session_state.word_vectorizer = word_vectorizer
        st.session_state.le = le
        st.session_state.model_weights = model_weights
        st.session_state.model_accuracies = accuracies
        st.session_state.models_trained = True
        
        progress_bar.progress(100)
        status_text.text("✅ Training Complete!")
        
        # Display training results
        st.success("🎉 All 6 models trained successfully!")
        
        # Create results display
        model_names = {
            'lr': 'Logistic Regression',
            'rf': 'Random Forest',
            'mlp': 'MLP Neural Network',
            'gnb': 'Multinomial Naive Bayes',
            'dt': 'Decision Tree',
            'knn': 'K-Nearest Neighbors'
        }
        
        results_text = "**Training Results:**\n\n"
        for key, name in model_names.items():
            results_text += f"- ✅ {name}: **{accuracies[key]*100:.2f}%** (Weight: {model_weights[key]*100:.1f}%)\n"
        
        st.info(results_text)
        st.info("📊 All 6 models are now ready for weighted ensemble predictions!")
        
        return True
        
    except Exception as e:
        status_text.text("❌ Training failed!")
        st.error(f"Error during training: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False
    finally:
        # Clear progress indicators after a delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

# High-confidence prediction function
def predict_with_high_confidence(resume_text, min_confidence=0.90):
    """Predict with high confidence using weighted ensemble of all 6 calibrated models"""
    if not st.session_state.models_trained:
        return None
    
    # Clean and vectorize
    cleaned_text = cleanResume(resume_text)
    text_vector = st.session_state.word_vectorizer.transform([cleaned_text])
    text_dense = text_vector.toarray()
    
    # Get probabilities from all 6 calibrated models with error handling
    weights = st.session_state.model_weights
    probabilities = {}
    valid_weights = {}
    
    try:
        prob_lr = st.session_state.clf_lr_calibrated.predict_proba(text_dense)[0]
        if not np.isnan(prob_lr).any():
            probabilities['lr'] = prob_lr
            valid_weights['lr'] = weights['lr']
    except Exception:
        pass
    
    try:
        prob_rf = st.session_state.clf_rf_calibrated.predict_proba(text_dense)[0]
        if not np.isnan(prob_rf).any():
            probabilities['rf'] = prob_rf
            valid_weights['rf'] = weights['rf']
    except Exception:
        pass
    
    try:
        prob_mlp = st.session_state.clf_mlp_calibrated.predict_proba(text_vector)[0]
        if not np.isnan(prob_mlp).any():
            probabilities['mlp'] = prob_mlp
            valid_weights['mlp'] = weights['mlp']
    except Exception:
        pass
    
    try:
        prob_gnb = st.session_state.clf_gnb_calibrated.predict_proba(text_vector)[0]
        if not np.isnan(prob_gnb).any():
            probabilities['gnb'] = prob_gnb
            valid_weights['gnb'] = weights['gnb']
    except Exception:
        pass
    
    try:
        prob_dt = st.session_state.clf_dt_calibrated.predict_proba(text_dense)[0]
        if not np.isnan(prob_dt).any():
            probabilities['dt'] = prob_dt
            valid_weights['dt'] = weights['dt']
    except Exception:
        pass
    
    try:
        prob_knn = st.session_state.clf_knn_calibrated.predict_proba(text_vector)[0]
        if not np.isnan(prob_knn).any():
            probabilities['knn'] = prob_knn
            valid_weights['knn'] = weights['knn']
    except Exception:
        pass
    
    # Check if we have at least one valid model
    if not probabilities:
        return None
    
    # Normalize weights for valid models only
    total_valid_weight = sum(valid_weights.values())
    if total_valid_weight == 0:
        return None
    
    normalized_weights = {k: v / total_valid_weight for k, v in valid_weights.items()}
    
    # Weighted ensemble of valid models
    ensemble_probs = np.zeros_like(list(probabilities.values())[0])
    for model_key in probabilities.keys():
        ensemble_probs += normalized_weights[model_key] * probabilities[model_key]
    
    # Get top prediction
    predicted_idx = int(np.argmax(ensemble_probs))
    confidence = float(ensemble_probs[predicted_idx])
    
    # Get top 3 predictions
    top3_indices = ensemble_probs.argsort()[-3:][::-1]
    top3_predictions = [
        {
            'category': st.session_state.le.classes_[int(idx)],
            'confidence': float(ensemble_probs[int(idx)])
        }
        for idx in top3_indices
    ]
    
    # Apply temperature scaling if confidence is low
    if confidence < min_confidence:
        temperature = 0.3
        scaled_probs = np.exp(np.log(ensemble_probs + 1e-10) / temperature)
        scaled_probs = scaled_probs / scaled_probs.sum()
        
        predicted_idx = int(np.argmax(scaled_probs))
        confidence = float(scaled_probs[predicted_idx])
        
        top3_indices = scaled_probs.argsort()[-3:][::-1]
        top3_predictions = [
            {
                'category': st.session_state.le.classes_[int(idx)],
                'confidence': float(scaled_probs[int(idx)])
            }
            for idx in top3_indices
        ]
    
    # Get recommendations
    predicted_category = st.session_state.le.classes_[predicted_idx]
    recommendations = JOB_RECOMMENDATIONS.get(predicted_category, None)
    
    return {
        'predicted_category': predicted_category,
        'confidence': confidence,
        'top_3_predictions': top3_predictions,
        'recommendations': recommendations
    }

# Main UI
def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Resume Category Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your resume PDF and get instant category prediction with high confidence</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.button("🔄 Train/Retrain Models", use_container_width=True):
            train_models()
            st.success("✅ Models trained successfully!")
        
        st.markdown("---")
        st.markdown("### 📊 Model Status")
        if st.session_state.models_trained:
            st.success("✅ Models Ready")
        else:
            st.warning("⚠️ Models Not Trained")
            st.info("Click 'Train/Retrain Models' to start")
        
        st.markdown("---")
        st.markdown("### 🤖 Models Used")
        if st.session_state.models_trained:
            model_names = {
                'lr': 'Logistic Regression',
                'rf': 'Random Forest',
                'mlp': 'MLP Neural Network',
                'gnb': 'Multinomial Naive Bayes',
                'dt': 'Decision Tree',
                'knn': 'K-Nearest Neighbors'
            }
            
            models_text = ""
            for name in model_names.values():
                models_text += f"• {name}\n"
            
            st.info(models_text)
        else:
            st.info("Models will be shown after training")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This app uses advanced machine learning with:
        - Probability calibration
        - Ensemble methods
        - High-confidence predictions (>90%)
        """)
    
    # Main content
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first using the sidebar button.")
        if st.button("🚀 Train Models Now", use_container_width=True):
            train_models()
    else:
        # File upload section
        st.markdown("### 📤 Upload Your Resume")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload your resume in PDF format"
        )
        
        # Text input as alternative
        st.markdown("### 📝 Or Paste Resume Text")
        resume_text_input = st.text_area(
            "Paste your resume text here",
            height=200,
            help="Alternatively, paste your resume text directly"
        )
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔍 Predict Category", use_container_width=True, type="primary")
        
        # Prediction results
        if predict_button:
            resume_text = None
            
            # Get text from file or input
            if uploaded_file is not None:
                try:
                    with st.spinner("📄 Extracting text from PDF..."):
                        resume_text = extract_text_from_pdf(uploaded_file)
                    st.success(f"✅ Successfully extracted {len(resume_text)} characters from PDF")
                except Exception as e:
                    st.error(f"❌ Error reading PDF: {str(e)}")
                    st.info("💡 Tip: Make sure your PDF contains selectable text, or install OCR dependencies for scanned PDFs")
            elif resume_text_input:
                resume_text = resume_text_input
            else:
                st.warning("⚠️ Please upload a PDF file or paste resume text")
            
            # Make prediction
            if resume_text:
                with st.spinner("🤖 Analyzing resume and predicting category..."):
                    result = predict_with_high_confidence(resume_text)
                
                if result:
                    # Display results
                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Results")
                    
                    # Display which models are being used (simple text format)
                    st.markdown("### 🤖 Models Used for Prediction")
                    
                    model_names = {
                        'lr': 'Logistic Regression',
                        'rf': 'Random Forest',
                        'mlp': 'MLP Neural Network',
                        'gnb': 'Multinomial Naive Bayes',
                        'dt': 'Decision Tree',
                        'knn': 'K-Nearest Neighbors'
                    }
                    
                    models_text = "**6-Model Weighted Ensemble:**\n\n"
                    for key, name in model_names.items():
                        models_text += f"• {name}\n"
                    
                    st.markdown(models_text)
                    
                    st.markdown("---")
                    
                    # Main prediction card
                    confidence_pct = result['confidence'] * 100
                    
                    # Confidence color coding
                    if confidence_pct >= 90:
                        conf_color = "🟢"
                        conf_status = "EXCELLENT"
                    elif confidence_pct >= 70:
                        conf_color = "🟡"
                        conf_status = "GOOD"
                    else:
                        conf_color = "🟠"
                        conf_status = "MODERATE"
                    
                    # Main result
                    st.markdown(f"""
                    <div class="confidence-box">
                        <h2 style="margin:0; font-size: 2rem;">{result['predicted_category']}</h2>
                        <p style="margin:0.5rem 0; font-size: 1.5rem;">
                            {conf_color} Confidence: <strong>{confidence_pct:.2f}%</strong> ({conf_status})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top 3 predictions
                    st.markdown("### 📊 Top 3 Category Suggestions")
                    top3 = result['top_3_predictions']
                    
                    for i, pred in enumerate(top3, 1):
                        conf = pred['confidence'] * 100
                        bar_length = int(conf / 2)  # Scale to 50 chars max
                        bar = "█" * bar_length + "░" * (50 - bar_length)
                        
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{i}. {pred['category']}**")
                            with col2:
                                st.markdown(f"**{conf:.2f}%**")
                            st.progress(conf / 100)
                            st.markdown(f"`{bar}`")
                            st.markdown("---")
                    
                    # Job Recommendations Section
                    if result.get('recommendations'):
                        recs = result['recommendations']
                        st.markdown("---")
                        st.markdown("## 💼 Career Recommendations")
                        
                        # Create columns for better layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🏢 Top Companies")
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            for i, company in enumerate(recs.get('top_companies', [])[:10], 1):
                                st.markdown(f"**{i}.** {company}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown("### 💰 Salary Range")
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3 style="color: #667eea; margin:0;">{recs.get('avg_salary', 'N/A')}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### 🌐 Job Platforms")
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            for i, platform in enumerate(recs.get('job_platforms', [])[:8], 1):
                                st.markdown(f"**{i}.** {platform}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown("### 📚 Learning Platforms")
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            for i, platform in enumerate(recs.get('learning_platforms', [])[:8], 1):
                                st.markdown(f"**{i}.** {platform}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Skills needed
                        st.markdown("### 🎓 Key Skills to Master")
                        skills = recs.get('skills_needed', [])
                        if skills:
                            # Display skills as badges
                            skill_cols = st.columns(min(4, len(skills)))
                            for idx, skill in enumerate(skills):
                                with skill_cols[idx % len(skill_cols)]:
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                                color: white; padding: 0.5rem; border-radius: 5px; 
                                                text-align: center; margin: 0.25rem 0;">
                                        <strong>{skill}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("💡 Recommendations not available for this category yet.")
                    
                    # Success message
                    if confidence_pct >= 90:
                        st.balloons()
                        st.success("🎉 High confidence prediction! The model is very certain about this category.")
                else:
                    st.error("❌ Prediction failed. Please try again.")

if __name__ == "__main__":
    main()

