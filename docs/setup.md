# 🛠️ Setup Guide

This guide provides detailed instructions for setting up the PredictFlex Customer Churn Prediction application on your local machine.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space
- **Internet**: Required for downloading dependencies

### Required Software
1. **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
2. **Git**: Download from [git-scm.com](https://git-scm.com/)
3. **pip**: Usually comes with Python installation

### Verify Installation
```bash
# Check Python version
python --version

# Check pip version
pip --version

# Check Git version
git --version
```

## 🚀 Installation Steps

### Step 1: Clone the Repository
```bash
# Clone the repository
git clone <repository-url>
cd predictflex

# Verify the project structure
ls -la
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### Step 4: Verify Installation
```bash
# Test Streamlit installation
streamlit --version

# Test Python imports
python -c "import streamlit, pandas, numpy, sklearn, xgboost; print('All dependencies installed successfully!')"
```

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file in the project root for custom configurations:
```bash
# .env file
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
MODEL_SAVE_PATH=models/
LOG_PATH=logs/
```

### Directory Structure Setup
The application will automatically create required directories:
```bash
# Create directories manually (if needed)
mkdir -p models
mkdir -p logs
mkdir -p data
```

## 🚀 Running the Application

### Basic Run
```bash
# Start the application
streamlit run app.py
```

### Advanced Run Options
```bash
# Run on specific port
streamlit run app.py --server.port 8502

# Run on specific address
streamlit run app.py --server.address 0.0.0.0

# Run with debug mode
streamlit run app.py --logger.level debug
```

### Access the Application
1. Open your web browser
2. Navigate to `http://localhost:8501`
3. The application should load automatically

## 🧪 Testing the Installation

### Test Data Upload
1. Create a sample CSV file with customer data:
```csv
customer_id,age,tenure,monthly_charges,total_charges,churn
1,25,12,29.85,358.2,0
2,45,24,56.95,1366.8,1
3,32,6,42.30,253.8,0
```

2. Upload the file in the application
3. Verify that the data loads correctly

### Test Model Training
1. Upload a dataset with a churn column
2. Select "Automatic (best accuracy)" model type
3. Click "Train Model"
4. Verify that training completes successfully

## 🔍 Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install streamlit specifically
pip install streamlit
```

#### Issue: "Permission denied" on Windows
**Solution:**
```bash
# Run PowerShell as Administrator
# Or use virtual environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Issue: "Port already in use"
**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
# On Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

#### Issue: "Memory error" with large datasets
**Solution:**
- Reduce dataset size
- Use data sampling in the application
- Increase system RAM
- Use cloud deployment for large datasets

#### Issue: "Model training fails"
**Solution:**
- Check data quality (no missing values in target column)
- Ensure target column is binary (0/1)
- Verify sufficient data (minimum 100 rows recommended)
- Check for data type issues

### Performance Optimization

#### For Large Datasets
```bash
# Set environment variables for better performance
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
```

#### For Development
```bash
# Enable debug mode
streamlit run app.py --logger.level debug

# Enable auto-reload
streamlit run app.py --server.runOnSave true
```

## 🌐 Deployment Options

### Local Development
- Use virtual environment
- Run on localhost
- Suitable for development and testing

### Cloud Deployment
- **Streamlit Cloud**: Free hosting for Streamlit apps
- **Heroku**: Platform as a Service
- **AWS/GCP/Azure**: Cloud infrastructure
- **Docker**: Containerized deployment

### Docker Deployment (Advanced)
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📊 System Monitoring

### Check Application Status
```bash
# Check if Streamlit is running
ps aux | grep streamlit

# Check port usage
netstat -tulpn | grep 8501
```

### Monitor Logs
```bash
# View application logs
tail -f logs/prediction_log.csv

# View Streamlit logs
streamlit run app.py --logger.level debug
```

## 🔒 Security Considerations

### Development Environment
- Use virtual environments
- Don't commit sensitive data
- Use environment variables for secrets

### Production Environment
- Use HTTPS
- Implement authentication
- Regular security updates
- Monitor access logs

## 📞 Getting Help

### Documentation
- Check this setup guide
- Review the main README
- Explore the API documentation

### Community Support
- GitHub Issues
- Stack Overflow
- Streamlit Community

### Debug Information
When reporting issues, include:
- Operating system and version
- Python version
- Error messages
- Steps to reproduce
- Sample data (if applicable)

---

**Next Steps**: After successful setup, proceed to [Usage Examples](examples.md) or [API Reference](api_reference.md). 