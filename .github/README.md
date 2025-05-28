# GitHub Actions CI/CD Pipeline Documentation

This directory contains the GitHub Actions workflows for the Energy Generation Dashboard project, providing comprehensive CI/CD automation including testing, security scanning, performance monitoring, and deployment.

## 🚀 Workflows Overview

### 1. CI/CD Pipeline (`ci.yml`)
**Triggers:** Push to main/develop, Pull Requests, Daily schedule
**Purpose:** Main continuous integration and deployment pipeline

**Jobs:**
- **Code Quality & Linting:** Black, isort, Flake8, MyPy, Bandit, Safety
- **Unit Tests:** Multi-Python version testing with coverage
- **Integration Tests:** End-to-end application testing
- **Performance Tests:** Benchmark and load testing
- **Security Scan:** Trivy vulnerability scanning
- **Docker Build:** Container image building and testing
- **Test Report:** Comprehensive test result generation
- **Deploy:** Automated deployment to staging/production
- **Notification:** Status notifications and PR comments

### 2. Release Pipeline (`release.yml`)
**Triggers:** Release published, Manual workflow dispatch
**Purpose:** Production release and deployment automation

**Jobs:**
- **Pre-release Validation:** Comprehensive test suite execution
- **Build Artifacts:** Python package and release archive creation
- **Docker Build & Push:** Multi-platform container images
- **Deploy to Staging:** Staging environment deployment
- **Staging Integration Tests:** Post-deployment validation
- **Deploy to Production:** Production deployment with approvals
- **Post-deployment Validation:** Health checks and monitoring
- **GitHub Release:** Automated release notes and asset uploads

### 3. Security & Maintenance (`security-and-maintenance.yml`)
**Triggers:** Daily/Weekly schedule, Manual dispatch
**Purpose:** Automated security monitoring and maintenance

**Jobs:**
- **Security Scan:** Daily vulnerability scanning (Safety, Bandit, Semgrep, Trivy)
- **Dependency Updates:** Weekly automated dependency updates
- **Performance Monitoring:** System performance tracking
- **Code Quality Monitoring:** Code complexity and maintainability analysis
- **Maintenance Report:** Comprehensive maintenance status reporting

### 4. Performance Testing (`performance.yml`)
**Triggers:** Push to main, Pull Requests, Weekly schedule, Manual dispatch
**Purpose:** Performance monitoring and regression detection

**Jobs:**
- **Performance Benchmark:** Execution time benchmarking
- **Memory Profiling:** Memory usage analysis
- **Load Testing:** Concurrent request simulation
- **Performance Regression:** Baseline comparison and alerts
- **Performance Report:** Comprehensive performance analysis

## 🔧 Setup Instructions

### 1. Repository Secrets
Configure the following secrets in your GitHub repository:

```
GITHUB_TOKEN          # Automatically provided by GitHub
PRESCINTO_API_TOKEN   # Your Prescinto API token (if using external API)
```

### 2. Environment Configuration
Set up GitHub Environments for deployment:

**Staging Environment:**
- Protection rules: None (auto-deploy)
- Environment secrets: Staging-specific configurations

**Production Environment:**
- Protection rules: Required reviewers
- Environment secrets: Production configurations

### 3. Branch Protection Rules
Configure branch protection for `main` and `develop`:
- Require status checks to pass
- Require branches to be up to date
- Require review from code owners
- Restrict pushes to specific people/teams

## 📊 Workflow Features

### Automated Testing
- **Multi-Python Version Support:** Tests run on Python 3.8, 3.9, 3.10, 3.11
- **Comprehensive Coverage:** Unit, integration, and performance tests
- **Coverage Reporting:** Codecov integration with detailed reports
- **Test Artifacts:** Downloadable test reports and logs

### Security Monitoring
- **Dependency Scanning:** Daily security vulnerability checks
- **Static Analysis:** Code security analysis with Bandit and Semgrep
- **Container Scanning:** Docker image vulnerability assessment
- **Automated Issues:** Security issues automatically created for critical findings

### Performance Monitoring
- **Benchmark Testing:** Automated performance benchmarking
- **Memory Profiling:** Memory usage analysis and leak detection
- **Load Testing:** Concurrent request simulation
- **Regression Detection:** Performance regression alerts on PRs

### Code Quality
- **Linting:** Flake8 for code style enforcement
- **Formatting:** Black for consistent code formatting
- **Import Sorting:** isort for organized imports
- **Type Checking:** MyPy for static type analysis
- **Complexity Analysis:** Radon for code complexity monitoring

### Deployment Automation
- **Multi-Environment:** Staging and production deployments
- **Docker Integration:** Containerized deployment support
- **Health Checks:** Post-deployment validation
- **Rollback Support:** Automated rollback on deployment failures

## 🎯 Usage Examples

### Running Specific Workflows

**Trigger Security Scan:**
```bash
gh workflow run security-and-maintenance.yml -f scan_type=security
```

**Trigger Performance Tests:**
```bash
gh workflow run performance.yml -f test_type=benchmark
```

**Manual Release:**
```bash
gh workflow run release.yml -f version=v1.2.0 -f environment=production
```

### Monitoring Workflow Status

**Check Workflow Runs:**
```bash
gh run list --workflow=ci.yml
```

**View Workflow Details:**
```bash
gh run view [RUN_ID]
```

**Download Artifacts:**
```bash
gh run download [RUN_ID]
```

## 📈 Metrics and Reporting

### Test Coverage
- **Target:** >80% overall coverage, >90% for critical modules
- **Reporting:** Codecov integration with PR comments
- **Trends:** Coverage trend monitoring over time

### Performance Metrics
- **Benchmarks:** Execution time tracking for critical functions
- **Memory Usage:** Memory consumption monitoring
- **Load Testing:** Concurrent request handling capacity

### Security Metrics
- **Vulnerability Count:** Number of security issues detected
- **Dependency Health:** Outdated and vulnerable dependency tracking
- **Code Security:** Static analysis security findings

## 🔍 Troubleshooting

### Common Issues

**Test Failures:**
1. Check test logs in workflow artifacts
2. Run tests locally: `python tests/test_runner.py --all`
3. Verify dependencies: `python tests/test_validation.py`

**Docker Build Failures:**
1. Check Dockerfile syntax
2. Verify base image availability
3. Review build logs for dependency issues

**Deployment Failures:**
1. Check environment configuration
2. Verify secrets and permissions
3. Review deployment logs

**Performance Regressions:**
1. Compare benchmark results with baseline
2. Identify performance bottlenecks
3. Optimize critical code paths

### Debug Commands

**Local Testing:**
```bash
# Run all tests locally
python tests/test_runner.py --all --verbose

# Run specific test category
python tests/test_runner.py --unit --coverage

# Validate test setup
python tests/test_validation.py

# Run with Docker
docker-compose --profile testing up test
```

**Security Scanning:**
```bash
# Run security scans locally
docker-compose --profile security up security-scan

# Manual security checks
safety check
bandit -r backend/ frontend/ src/
```

## 📚 Best Practices

### Workflow Optimization
1. **Cache Dependencies:** Use GitHub Actions cache for pip dependencies
2. **Parallel Execution:** Run independent jobs in parallel
3. **Conditional Execution:** Use conditions to skip unnecessary jobs
4. **Artifact Management:** Clean up old artifacts regularly

### Security Best Practices
1. **Secret Management:** Use GitHub secrets for sensitive data
2. **Least Privilege:** Grant minimum required permissions
3. **Regular Updates:** Keep actions and dependencies updated
4. **Audit Logs:** Monitor workflow execution logs

### Performance Optimization
1. **Resource Limits:** Set appropriate resource limits for jobs
2. **Timeout Configuration:** Configure reasonable timeouts
3. **Efficient Testing:** Optimize test execution time
4. **Selective Testing:** Run relevant tests based on changes

## 🔄 Maintenance

### Regular Tasks
- **Weekly:** Review security scan results
- **Monthly:** Update GitHub Actions versions
- **Quarterly:** Review and optimize workflow performance
- **As Needed:** Update deployment configurations

### Monitoring
- **Workflow Success Rate:** Monitor overall pipeline health
- **Execution Time:** Track workflow execution duration
- **Resource Usage:** Monitor GitHub Actions usage limits
- **Cost Optimization:** Optimize for GitHub Actions billing

## 📞 Support

For issues with the CI/CD pipeline:
1. Check workflow logs and artifacts
2. Review this documentation
3. Create an issue with workflow details
4. Contact the development team

---

*This CI/CD pipeline is designed to ensure code quality, security, and reliable deployments for the Energy Generation Dashboard project.*
