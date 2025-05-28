#!/bin/bash

# CI/CD Setup Script for Energy Generation Dashboard
# This script helps set up the GitHub Actions CI/CD pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GitHub CLI authentication
check_gh_auth() {
    if command_exists gh; then
        if gh auth status >/dev/null 2>&1; then
            print_status "GitHub CLI is authenticated"
            return 0
        else
            print_warning "GitHub CLI is not authenticated"
            return 1
        fi
    else
        print_warning "GitHub CLI is not installed"
        return 1
    fi
}

# Function to setup repository secrets
setup_secrets() {
    print_header "Setting up repository secrets..."
    
    if ! check_gh_auth; then
        print_error "GitHub CLI authentication required for secret setup"
        print_status "Please run: gh auth login"
        return 1
    fi
    
    # Check if secrets already exist
    print_status "Checking existing secrets..."
    
    # PRESCINTO_API_TOKEN (if using external API)
    read -p "Do you want to set up PRESCINTO_API_TOKEN? (y/n): " setup_api_token
    if [[ $setup_api_token == "y" || $setup_api_token == "Y" ]]; then
        read -s -p "Enter PRESCINTO_API_TOKEN: " api_token
        echo
        if [[ -n "$api_token" ]]; then
            echo "$api_token" | gh secret set PRESCINTO_API_TOKEN
            print_status "PRESCINTO_API_TOKEN secret set successfully"
        else
            print_warning "Empty token provided, skipping..."
        fi
    fi
    
    print_status "Repository secrets setup completed"
}

# Function to setup environments
setup_environments() {
    print_header "Setting up GitHub environments..."
    
    if ! check_gh_auth; then
        print_error "GitHub CLI authentication required for environment setup"
        return 1
    fi
    
    # Create staging environment
    print_status "Creating staging environment..."
    gh api repos/:owner/:repo/environments/staging --method PUT --field prevent_self_review=false
    
    # Create production environment with protection rules
    print_status "Creating production environment with protection rules..."
    gh api repos/:owner/:repo/environments/production --method PUT --field prevent_self_review=true
    
    # Add required reviewers for production (you'll need to modify this)
    read -p "Enter GitHub username for production environment reviewer: " reviewer
    if [[ -n "$reviewer" ]]; then
        gh api repos/:owner/:repo/environments/production --method PUT \
            --field "reviewers[0][type]=User" \
            --field "reviewers[0][id]=$reviewer"
        print_status "Production environment reviewer set to: $reviewer"
    fi
    
    print_status "GitHub environments setup completed"
}

# Function to setup branch protection
setup_branch_protection() {
    print_header "Setting up branch protection rules..."
    
    if ! check_gh_auth; then
        print_error "GitHub CLI authentication required for branch protection setup"
        return 1
    fi
    
    # Protect main branch
    print_status "Setting up protection for main branch..."
    gh api repos/:owner/:repo/branches/main/protection --method PUT \
        --field "required_status_checks[strict]=true" \
        --field "required_status_checks[contexts][0]=Code Quality & Linting" \
        --field "required_status_checks[contexts][1]=Unit Tests (3.9)" \
        --field "required_status_checks[contexts][2]=Integration Tests" \
        --field "enforce_admins=false" \
        --field "required_pull_request_reviews[required_approving_review_count]=1" \
        --field "required_pull_request_reviews[dismiss_stale_reviews]=true" \
        --field "restrictions=null"
    
    # Protect develop branch (if it exists)
    if gh api repos/:owner/:repo/branches/develop >/dev/null 2>&1; then
        print_status "Setting up protection for develop branch..."
        gh api repos/:owner/:repo/branches/develop/protection --method PUT \
            --field "required_status_checks[strict]=true" \
            --field "required_status_checks[contexts][0]=Unit Tests (3.9)" \
            --field "enforce_admins=false" \
            --field "required_pull_request_reviews[required_approving_review_count]=1" \
            --field "restrictions=null"
    fi
    
    print_status "Branch protection rules setup completed"
}

# Function to validate workflow files
validate_workflows() {
    print_header "Validating workflow files..."
    
    workflow_dir=".github/workflows"
    
    if [[ ! -d "$workflow_dir" ]]; then
        print_error "Workflow directory not found: $workflow_dir"
        return 1
    fi
    
    # Check for required workflow files
    required_workflows=("ci.yml" "release.yml" "security-and-maintenance.yml" "performance.yml")
    
    for workflow in "${required_workflows[@]}"; do
        if [[ -f "$workflow_dir/$workflow" ]]; then
            print_status "✓ $workflow found"
        else
            print_error "✗ $workflow not found"
        fi
    done
    
    # Validate YAML syntax (if yamllint is available)
    if command_exists yamllint; then
        print_status "Validating YAML syntax..."
        for workflow_file in "$workflow_dir"/*.yml; do
            if yamllint "$workflow_file" >/dev/null 2>&1; then
                print_status "✓ $(basename "$workflow_file") syntax valid"
            else
                print_warning "✗ $(basename "$workflow_file") syntax issues detected"
            fi
        done
    else
        print_warning "yamllint not available, skipping YAML validation"
    fi
    
    print_status "Workflow validation completed"
}

# Function to setup local development environment
setup_local_dev() {
    print_header "Setting up local development environment..."
    
    # Check if Docker is available
    if command_exists docker; then
        print_status "Docker is available"
        
        # Check if Docker Compose is available
        if command_exists docker-compose; then
            print_status "Docker Compose is available"
            
            # Build development image
            print_status "Building development Docker image..."
            docker-compose build app
            
            print_status "Testing Docker setup..."
            docker-compose --profile testing up --build test-unit --exit-code-from test-unit
            
        else
            print_warning "Docker Compose not available"
        fi
    else
        print_warning "Docker not available"
    fi
    
    # Setup Python virtual environment
    if command_exists python3; then
        print_status "Setting up Python virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements-test.txt
        
        print_status "Running test validation..."
        python tests/test_validation.py
        
        print_status "Local development environment setup completed"
    else
        print_error "Python 3 not available"
    fi
}

# Function to create initial commit with CI/CD setup
create_initial_commit() {
    print_header "Creating initial CI/CD commit..."
    
    if command_exists git; then
        # Check if we're in a git repository
        if git rev-parse --git-dir >/dev/null 2>&1; then
            print_status "Adding CI/CD files to git..."
            git add .github/ scripts/ Dockerfile docker-compose.yml pytest.ini requirements-test.txt
            
            if git diff --cached --quiet; then
                print_warning "No changes to commit"
            else
                git commit -m "feat: add comprehensive CI/CD pipeline with GitHub Actions

- Add CI/CD pipeline with testing, security, and deployment
- Add performance monitoring and regression detection
- Add security scanning and dependency updates
- Add Docker support for containerized deployment
- Add comprehensive test suite with coverage reporting"
                
                print_status "CI/CD setup committed successfully"
                
                # Ask if user wants to push
                read -p "Do you want to push the changes to remote? (y/n): " push_changes
                if [[ $push_changes == "y" || $push_changes == "Y" ]]; then
                    git push
                    print_status "Changes pushed to remote repository"
                fi
            fi
        else
            print_warning "Not in a git repository, skipping commit"
        fi
    else
        print_warning "Git not available, skipping commit"
    fi
}

# Function to display setup summary
display_summary() {
    print_header "CI/CD Setup Summary"
    echo
    echo "✅ GitHub Actions workflows configured:"
    echo "   - ci.yml: Main CI/CD pipeline"
    echo "   - release.yml: Release and deployment automation"
    echo "   - security-and-maintenance.yml: Security monitoring"
    echo "   - performance.yml: Performance testing"
    echo
    echo "✅ Docker configuration:"
    echo "   - Dockerfile: Multi-stage container builds"
    echo "   - docker-compose.yml: Local development environment"
    echo
    echo "✅ Testing framework:"
    echo "   - pytest.ini: Test configuration"
    echo "   - requirements-test.txt: Test dependencies"
    echo "   - Comprehensive test suite with coverage"
    echo
    echo "🔧 Next steps:"
    echo "   1. Push changes to trigger first CI/CD run"
    echo "   2. Review workflow results in GitHub Actions tab"
    echo "   3. Configure any additional secrets or environments"
    echo "   4. Set up monitoring and alerting as needed"
    echo
    print_status "CI/CD setup completed successfully!"
}

# Main setup function
main() {
    print_header "Energy Dashboard CI/CD Setup"
    echo "This script will help you set up the GitHub Actions CI/CD pipeline."
    echo
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    # Validate workflow files
    validate_workflows
    
    # Setup options
    echo
    echo "Setup options:"
    echo "1. Repository secrets"
    echo "2. GitHub environments"
    echo "3. Branch protection"
    echo "4. Local development environment"
    echo "5. All of the above"
    echo
    
    read -p "Select setup option (1-5): " setup_option
    
    case $setup_option in
        1)
            setup_secrets
            ;;
        2)
            setup_environments
            ;;
        3)
            setup_branch_protection
            ;;
        4)
            setup_local_dev
            ;;
        5)
            setup_secrets
            setup_environments
            setup_branch_protection
            setup_local_dev
            create_initial_commit
            ;;
        *)
            print_error "Invalid option selected"
            exit 1
            ;;
    esac
    
    display_summary
}

# Run main function
main "$@"
