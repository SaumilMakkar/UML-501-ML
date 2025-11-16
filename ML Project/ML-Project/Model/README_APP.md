# 📄 Resume Category Predictor - Streamlit App

A beautiful, modern web frontend for resume categorization with high-confidence predictions (>90%).

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

Or install Streamlit separately:
```bash
pip install streamlit PyPDF2 pdfplumber pdf2image pytesseract Pillow
```

### 2. Run the App

Navigate to the Model directory and run:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Features

✨ **Beautiful Modern UI**
- Gradient headers and styled components
- Responsive design
- Visual confidence indicators

📤 **Multiple Input Methods**
- PDF file upload
- Direct text paste
- OCR support for scanned PDFs

🎯 **High-Confidence Predictions**
- Uses calibrated ensemble models
- Temperature scaling for >90% confidence
- Top 3 category suggestions with visual bars

⚙️ **Easy Model Management**
- One-click model training
- Model status indicator
- Automatic caching

## 🎨 UI Components

1. **Main Header**:**
   - Gradient title with icon
   - Subtitle explaining the app

2. **Sidebar:**
   - Train/Retrain Models button
   - Model status indicator
   - About section

3. **Upload Section:**
   - PDF file uploader
   - Text area for direct input
   - Large, prominent predict button

4. **Results Display:**
   - Large confidence box with gradient
   - Top 3 predictions with progress bars
   - Visual confidence indicators (🟢🟡🟠)
   - Celebration animation for high confidence

## 🔧 Configuration

### Dataset Path
The app looks for the dataset at:
- `../Dataset/UpdatedResumeDataSet.csv` (default)
- `Dataset/UpdatedResumeDataSet.csv` (fallback)

### OCR Support
For scanned PDFs, ensure:
1. Tesseract OCR is installed
2. Path is set in environment variable `TESSERACT_CMD` or default location

## 📦 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Point to `app.py` file
4. Deploy!

### Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🎯 Usage

1. **First Time:**
   - Click "Train/Retrain Models" in sidebar
   - Wait for training to complete (2-5 minutes)

2. **Predict Category:**
   - Upload PDF or paste text
   - Click "Predict Category"
   - View results with confidence scores

3. **Results:**
   - Main predicted category with confidence
   - Top 3 suggestions with percentages
   - Visual progress bars

## 🐛 Troubleshooting

**Models not training:**
- Check dataset path
- Ensure all dependencies are installed
- Check console for error messages

**PDF not reading:**
- Ensure PDF has selectable text
- Install OCR dependencies for scanned PDFs
- Check file size (large files may take time)

**Low confidence:**
- Ensure resume text is clear and complete
- Try pasting text directly instead of PDF
- Check if resume matches training categories

## 📝 Notes

- Models are cached in session state
- First prediction may take longer (model loading)
- OCR is optional but recommended for scanned PDFs
- Confidence scores are calibrated for accuracy

## 🎉 Enjoy!

Your beautiful resume categorization app is ready to use!

