# GitHub Actions Deployment Guide

## 🚀 Automated Deployment with GitHub Actions

This project includes a complete CI/CD pipeline using GitHub Actions for automated testing, building, and deployment.

---

## Prerequisites

1. **GitHub Repository**
   - Push your code to GitHub
   - Make sure you're on the `main` or `master` branch

2. **GitHub Secrets** (Required for deployment)
   
   Go to your repository → Settings → Secrets and variables → Actions → New repository secret

   Add the following secrets:

   ### For Docker Hub Deployment:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub password/token

   ### For Heroku Deployment:
   - `HEROKU_API_KEY`: Your Heroku API key
   - `HEROKU_APP_NAME`: Your Heroku app name
   - `HEROKU_EMAIL`: Your Heroku account email

---

## What the CI/CD Pipeline Does

### 1. **Automated Testing** (Runs on every push/PR)
   - ✅ Sets up Python environment
   - ✅ Installs dependencies
   - ✅ Runs unit tests (if available)
   - ✅ Verifies model architecture can be created

### 2. **Code Quality Checks**
   - ✅ Black (code formatting)
   - ✅ isort (import sorting)
   - ✅ Flake8 (linting)

### 3. **Docker Build**
   - ✅ Builds Docker image
   - ✅ Pushes to Docker Hub (on main branch)
   - ✅ Tags with `latest` and commit SHA

### 4. **Deployment Options**
   - ✅ Streamlit Cloud (auto-deploy)
   - ✅ Heroku (containerized)
   - ✅ Docker Hub (for manual deployment)

---

## Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Connect your GitHub repository
3. Set main file: `src/deployment/app.py`
4. Deploy!

**Pros:**
- ✅ Free tier available
- ✅ Auto-deploys on git push
- ✅ No configuration needed

**Cons:**
- ⚠️ Resource limits on free tier
- ⚠️ Model file size limits

### Option 2: Heroku (Recommended for Production)

**Setup:**
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create your-app-name

# Get API key
heroku auth:token

# Add to GitHub Secrets
# HEROKU_API_KEY, HEROKU_APP_NAME, HEROKU_EMAIL
```

**Deploy:**
- Push to main branch → GitHub Actions automatically deploys
- Or manually: `git push heroku main`

**Pros:**
- ✅ Production-ready
- ✅ Good free tier
- ✅ Easy scaling
- ✅ Custom domains

**Cons:**
- ⚠️ Requires Heroku account
- ⚠️ Sleep mode on free tier

### Option 3: Docker Hub + Your Server

**Build and run locally:**
```bash
# Build
docker build -t face-mask-detection .

# Run
docker run -p 8501:8501 face-mask-detection

# Or use Docker Compose
docker-compose up
```

**Deploy to your server:**
```bash
# Pull from Docker Hub
docker pull yourusername/face-mask-detection:latest

# Run
docker run -d -p 8501:8501 yourusername/face-mask-detection:latest
```

**Pros:**
- ✅ Full control
- ✅ No vendor lock-in
- ✅ Can use any cloud provider

**Cons:**
- ⚠️ Requires server management
- ⚠️ Need to handle SSL, domains, etc.

### Option 4: AWS/GCP/Azure

Use the Docker image with:
- **AWS**: Elastic Container Service (ECS) or Elastic Beanstalk
- **GCP**: Cloud Run or App Engine
- **Azure**: Container Instances or App Service

---

## Manual Deployment Steps

### 1. Prepare Your Repository

```bash
# Initialize git (if not already)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: Face mask detection system"

# Add remote
git remote add origin https://github.com/yourusername/face-mask-detection.git

# Push
git push -u origin main
```

### 2. Set Up Secrets

1. Go to your GitHub repo
2. Settings → Secrets and variables → Actions
3. Add required secrets (see Prerequisites)

### 3. Trigger Deployment

```bash
# Make any change and push
git add .
git commit -m "Trigger deployment"
git push
```

GitHub Actions will automatically:
1. Run tests
2. Build Docker image
3. Deploy (if on main branch)

---

## Monitoring Deployments

### GitHub Actions
- Go to your repo → Actions tab
- View workflow runs
- Check logs for any errors

### Streamlit Cloud
- Dashboard: [share.streamlit.io](https://share.streamlit.io/)
- View logs and metrics
- Manage deployments

### Heroku
```bash
# View logs
heroku logs --tail --app your-app-name

# Check status
heroku ps --app your-app-name

# Open app
heroku open --app your-app-name
```

### Docker
```bash
# View running containers
docker ps

# View logs
docker logs container-id

# Stop container
docker stop container-id
```

---

## Environment Variables

For sensitive data, use environment variables:

**GitHub Actions:**
- Add to repository secrets

**Heroku:**
```bash
heroku config:set VARIABLE_NAME=value --app your-app-name
```

**Docker:**
```bash
docker run -e VARIABLE_NAME=value ...
```

**Docker Compose:**
```yaml
environment:
  - VARIABLE_NAME=value
```

---

## Troubleshooting

### Build Failures

**Issue:** Docker build fails  
**Solution:** Check Dockerfile, ensure all dependencies in requirements.txt

**Issue:** Tests fail  
**Solution:** Run tests locally first: `pytest tests/`

### Deployment Failures

**Issue:** Heroku deployment fails  
**Solution:** Check Heroku logs: `heroku logs --tail`

**Issue:** Streamlit Cloud timeout  
**Solution:** Model file might be too large, use Git LFS

### Runtime Errors

**Issue:** Model not found  
**Solution:** Ensure `models/saved_model/` is included or downloaded at runtime

**Issue:** Out of memory  
**Solution:** Reduce batch size, use smaller model, or upgrade tier

---

## Performance Optimization

### 1. Model Size
```bash
# Quantize model for faster loading
python src/deployment/quantize.py --model_path models/saved_model --output models/tflite/model.tflite
```

### 2. Caching
Add to Streamlit app:
```python
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/saved_model')
```

### 3. Docker Image Size
Use multi-stage builds in Dockerfile (already optimized)

---

## Continuous Deployment Workflow

```
Developer Push → GitHub
       ↓
GitHub Actions Triggered
       ↓
   Tests Run
       ↓
  Tests Pass?
   ↓        ↓
  Yes       No → Notify Developer
   ↓
Build Docker Image
   ↓
Push to Docker Hub
   ↓
Deploy to Platform
   ↓
  Live!
```

---

## Cost Considerations

| Platform | Free Tier | Paid Plans |
|----------|-----------|------------|
| Streamlit Cloud | 1 private app | $20/month for more |
| Heroku | 550 dyno hours/month | $7/month hobby |
| Docker Hub | Unlimited public images | $5/month for private |
| AWS/GCP/Azure | Free tier available | Pay as you go |

---

## Security Best Practices

1. ✅ Never commit secrets to Git
2. ✅ Use environment variables for sensitive data
3. ✅ Enable HTTPS on production
4. ✅ Regularly update dependencies
5. ✅ Use `.dockerignore` to exclude sensitive files
6. ✅ Enable branch protection on main
7. ✅ Review pull requests before merging

---

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Set up GitHub secrets
3. ✅ Choose deployment platform
4. ✅ Configure platform-specific settings
5. ✅ Push to main branch to trigger deployment
6. ✅ Monitor deployment in Actions tab
7. ✅ Test deployed application
8. ✅ Set up custom domain (optional)

---

## Support & Resources

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Docker Docs**: https://docs.docker.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Heroku Docs**: https://devcenter.heroku.com/

---

**Status**: ✅ CI/CD Pipeline Ready  
**Last Updated**: 2025-12-25
