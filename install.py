# install.py
import subpackage
import sys

packages = [
    "flask==2.3.3",
    "flask-cors==4.0.0", 
    "flask-login==0.6.2",
    "flask-limiter==3.3.1",
    "numpy==1.24.3",
    "scikit-learn==1.3.0",
    "scipy==1.10.1",
    "redis==5.0.0",
    "celery==5.3.1",
    "gunicorn==21.2.0",
    "python-dotenv==1.0.0"
]

print("📦 در حال نصب کتابخانه‌ها...")
for package in packages:
    print(f"   نصب {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
print("✅ نصب کامل شد!")
