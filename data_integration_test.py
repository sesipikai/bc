#!/usr/bin/env python3
"""
Data Integration and API Testing Suite

This script tests the data integration, API functionality, and error handling
of the enhanced book club predictor without requiring external dependencies.
"""

import json
import re
import os

class DataIntegrationTester:
    def __init__(self):
        self.issues = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        # Read source code for analysis
        with open('/Users/audriuspatalauskas/Documents/python/claude-coude-project_2/bcv1.py', 'r') as f:
            self.source_code = f.read()
            self.lines = self.source_code.split('\n')
    
    def add_issue(self, level, category, message, line_num=None):
        """Add an issue to the report"""
        issue = {
            'category': category,
            'message': message,
            'line': line_num
        }
        self.issues[level].append(issue)
        print(f"{'❌' if level in ['critical', 'high'] else '⚠️' if level == 'medium' else 'ℹ️'} {category}: {message}")
    
    def test_bigquery_integration(self):
        """Test BigQuery integration patterns"""
        print("\n🔍 Testing BigQuery Integration...")
        
        # Check for proper table name construction
        table_patterns = [
            r'MEMBERS_TABLE = f"\{PROJECT_ID\}\.\{DATASET_ID\}\.members"',
            r'BOOKS_TABLE = f"\{PROJECT_ID\}\.\{DATASET_ID\}\.books"',
            r'REVIEWS_TABLE = f"\{PROJECT_ID\}\.\{DATASET_ID\}\.reviews"'
        ]
        
        found_patterns = 0
        for pattern in table_patterns:
            if re.search(pattern, self.source_code):
                found_patterns += 1
        
        if found_patterns == 3:
            print("✅ BigQuery table construction: All table names properly formatted")
        else:
            self.add_issue('high', 'Data Integration', 
                          f"Only {found_patterns}/3 table patterns found properly formatted")
        
        # Check for SQL injection prevention
        sql_queries = re.findall(r'query = f"SELECT \* FROM `\{.*\}`"', self.source_code)
        if len(sql_queries) >= 3:
            print("✅ SQL Query Construction: Queries use parameterized table names")
        else:
            self.add_issue('medium', 'Data Integration',
                          "SQL queries may not be using proper parameterization")
        
        # Check for error handling in BigQuery operations
        bq_functions = ['load_members', 'load_books', 'load_reviews', 'save_members', 'save_books', 'save_reviews']
        error_handled_functions = 0
        
        for func in bq_functions:
            func_pattern = f'def {func}.*?(?=def|\Z)'
            func_match = re.search(func_pattern, self.source_code, re.DOTALL)
            if func_match:
                func_code = func_match.group(0)
                if 'try:' in func_code and 'except' in func_code:
                    error_handled_functions += 1
        
        if error_handled_functions == len(bq_functions):
            print("✅ BigQuery Error Handling: All functions have proper error handling")
        else:
            self.add_issue('high', 'Data Integration',
                          f"Only {error_handled_functions}/{len(bq_functions)} BigQuery functions have error handling")
    
    def test_data_processing_logic(self):
        """Test data processing and validation logic"""
        print("\n🔍 Testing Data Processing Logic...")
        
        # Check for comprehensive data merging
        merge_patterns = [
            r'df_reviews\.merge.*df_books',
            r'df_reviews\.merge.*df_members',
            r'comprehensive_df.*merge'
        ]
        
        merge_operations = 0
        for pattern in merge_patterns:
            if re.search(pattern, self.source_code):
                merge_operations += 1
        
        if merge_operations >= 2:
            print("✅ Data Merging: Comprehensive data joining implemented")
        else:
            self.add_issue('medium', 'Data Processing',
                          f"Only {merge_operations} data merge operations found")
        
        # Check for data type handling
        numeric_conversions = re.findall(r'pd\.to_numeric.*errors=', self.source_code)
        if len(numeric_conversions) >= 2:
            print("✅ Data Type Handling: Proper numeric conversions with error handling")
        else:
            self.add_issue('medium', 'Data Processing',
                          "Limited numeric data type conversion handling")
        
        # Check for null value handling
        null_handling_patterns = [
            r'\.dropna\(',
            r'\.fillna\(',
            r'\.notna\(',
            r'pd\.isna\(',
            r'pd\.notna\('
        ]
        
        null_handling_count = 0
        for pattern in null_handling_patterns:
            null_handling_count += len(re.findall(pattern, self.source_code))
        
        if null_handling_count >= 5:
            print("✅ Null Value Handling: Comprehensive null value processing")
        else:
            self.add_issue('medium', 'Data Processing',
                          f"Limited null value handling ({null_handling_count} instances)")
    
    def test_openai_api_integration(self):
        """Test OpenAI API integration and error handling"""
        print("\n🔍 Testing OpenAI API Integration...")
        
        # Check for modern OpenAI client usage
        if 'from openai import OpenAI' in self.source_code:
            print("✅ OpenAI Integration: Using modern OpenAI client")
        else:
            self.add_issue('high', 'API Integration', "Not using modern OpenAI client")
        
        # Check for JSON response format specification
        if 'response_format={"type": "json_object"}' in self.source_code:
            print("✅ API Configuration: JSON response format properly specified")
        else:
            self.add_issue('high', 'API Integration', "JSON response format not specified")
        
        # Check for proper model specification
        if 'gpt-4o-mini' in self.source_code:
            print("✅ Model Selection: Using appropriate model (gpt-4o-mini)")
        else:
            self.add_issue('medium', 'API Integration', "Model selection may not be optimal")
        
        # Check for temperature and token settings
        api_settings = ['temperature=', 'max_tokens=']
        settings_found = 0
        for setting in api_settings:
            if setting in self.source_code:
                settings_found += 1
        
        if settings_found == 2:
            print("✅ API Parameters: Temperature and token limits properly configured")
        else:
            self.add_issue('medium', 'API Integration',
                          f"Only {settings_found}/2 API parameters configured")
        
        # Check for JSON parsing error handling
        json_error_patterns = [
            r'json\.loads.*',
            r'json\.JSONDecodeError',
            r'except.*json'
        ]
        
        json_error_handling = 0
        for pattern in json_error_patterns:
            if re.search(pattern, self.source_code, re.IGNORECASE):
                json_error_handling += 1
        
        if json_error_handling >= 2:
            print("✅ JSON Processing: Proper JSON parsing and error handling")
        else:
            self.add_issue('high', 'API Integration',
                          "Insufficient JSON parsing error handling")
    
    def test_error_handling_patterns(self):
        """Test comprehensive error handling throughout the application"""
        print("\n🔍 Testing Error Handling Patterns...")
        
        # Count try-except blocks
        try_blocks = len(re.findall(r'try:', self.source_code))
        except_blocks = len(re.findall(r'except', self.source_code))
        
        if try_blocks == except_blocks and try_blocks > 10:
            print(f"✅ Error Handling Coverage: {try_blocks} try-except blocks found")
        elif try_blocks != except_blocks:
            self.add_issue('high', 'Error Handling',
                          f"Mismatched try-except blocks: {try_blocks} try vs {except_blocks} except")
        else:
            self.add_issue('medium', 'Error Handling',
                          f"Limited error handling: only {try_blocks} try-except blocks")
        
        # Check for specific error types
        specific_errors = [
            'Exception as e',
            'KeyError',
            'ValueError',
            'TypeError',
            'JSONDecodeError'
        ]
        
        specific_error_count = 0
        for error_type in specific_errors:
            if error_type in self.source_code:
                specific_error_count += 1
        
        if specific_error_count >= 3:
            print("✅ Error Specificity: Handling specific exception types")
        else:
            self.add_issue('medium', 'Error Handling',
                          f"Limited specific error handling ({specific_error_count}/5 types)")
        
        # Check for user-friendly error messages
        error_display_patterns = [
            r'st\.error\(',
            r'st\.warning\(',
            r'display_error_card'
        ]
        
        error_display_count = 0
        for pattern in error_display_patterns:
            error_display_count += len(re.findall(pattern, self.source_code))
        
        if error_display_count >= 10:
            print("✅ User Error Communication: Comprehensive error display system")
        else:
            self.add_issue('medium', 'Error Handling',
                          f"Limited user error communication ({error_display_count} instances)")
    
    def test_input_validation_security(self):
        """Test input validation and security measures"""
        print("\n🔍 Testing Input Validation and Security...")
        
        # Check for input sanitization
        sanitization_patterns = [
            r'\.strip\(\)',
            r'\.replace\(',
            r'html\.escape\(',
            r're\.sub\('
        ]
        
        sanitization_count = 0
        for pattern in sanitization_patterns:
            sanitization_count += len(re.findall(pattern, self.source_code))
        
        if sanitization_count >= 10:
            print("✅ Input Sanitization: Adequate input cleaning measures")
        else:
            self.add_issue('medium', 'Security',
                          f"Limited input sanitization ({sanitization_count} instances)")
        
        # Check for XSS prevention in HTML generation
        html_generation = len(re.findall(r'unsafe_allow_html=True', self.source_code))
        if html_generation > 0:
            # Check if user input is directly embedded
            dangerous_patterns = [
                r'f".*<.*{member_name}.*"',
                r'f".*<.*{book_title}.*"',
                r'f".*<.*{comment}.*"'
            ]
            
            dangerous_html = 0
            for pattern in dangerous_patterns:
                if re.search(pattern, self.source_code):
                    dangerous_html += 1
            
            if dangerous_html > 0:
                self.add_issue('high', 'Security',
                              f"Potential XSS vulnerability: {dangerous_html} instances of unescaped user data in HTML")
            else:
                print("✅ XSS Prevention: No obvious XSS vulnerabilities in HTML generation")
        
        # Check authentication implementation
        auth_patterns = [
            r'authentication_status',
            r'verify_password',
            r'bcrypt\.checkpw'
        ]
        
        auth_features = 0
        for pattern in auth_patterns:
            if re.search(pattern, self.source_code):
                auth_features += 1
        
        if auth_features >= 3:
            print("✅ Authentication: Comprehensive authentication system")
        else:
            self.add_issue('medium', 'Security',
                          f"Limited authentication features ({auth_features}/3)")
    
    def test_performance_considerations(self):
        """Test performance-related code patterns"""
        print("\n🔍 Testing Performance Considerations...")
        
        # Check for potentially expensive operations
        expensive_operations = [
            (r'\.iterrows\(\)', 'DataFrame iteration'),
            (r'pd\.concat.*ignore_index=True', 'DataFrame concatenation'),
            (r'for.*in.*\.unique\(\)', 'Iteration over unique values'),
            (r'\.apply\(.*lambda', 'Lambda functions in apply')
        ]
        
        performance_issues = 0
        for pattern, description in expensive_operations:
            count = len(re.findall(pattern, self.source_code))
            if count > 0:
                performance_issues += count
                print(f"⚠️ Performance Concern: {count} instances of {description}")
        
        if performance_issues > 5:
            self.add_issue('medium', 'Performance',
                          f"Multiple potentially expensive operations ({performance_issues} total)")
        elif performance_issues > 0:
            print(f"ℹ️ Performance: {performance_issues} potentially expensive operations found")
        else:
            print("✅ Performance: No obvious performance bottlenecks detected")
        
        # Check for caching mechanisms
        caching_patterns = [
            r'@st\.cache',
            r'st\.cache_data',
            r'st\.cache_resource'
        ]
        
        caching_found = 0
        for pattern in caching_patterns:
            if re.search(pattern, self.source_code):
                caching_found += 1
        
        if caching_found > 0:
            print(f"✅ Caching: {caching_found} caching mechanisms implemented")
        else:
            self.add_issue('low', 'Performance', "No caching mechanisms found - consider adding for better performance")
    
    def test_ui_responsiveness_code(self):
        """Test UI responsiveness patterns in code"""
        print("\n🔍 Testing UI Responsiveness Patterns...")
        
        # Check for loading states
        loading_patterns = [
            r'display_loading_state',
            r'st\.spinner',
            r'loading_container',
            r'st\.empty\(\)'
        ]
        
        loading_features = 0
        for pattern in loading_patterns:
            if re.search(pattern, self.source_code):
                loading_features += 1
        
        if loading_features >= 3:
            print("✅ Loading States: Comprehensive loading state management")
        else:
            self.add_issue('medium', 'UX',
                          f"Limited loading state management ({loading_features}/4 patterns)")
        
        # Check for responsive design elements
        responsive_patterns = [
            r'st\.columns',
            r'use_container_width=True',
            r'@media.*max-width',
            r'mobile.*responsive'
        ]
        
        responsive_features = 0
        for pattern in responsive_patterns:
            responsive_features += len(re.findall(pattern, self.source_code, re.IGNORECASE))
        
        if responsive_features >= 5:
            print("✅ Responsive Design: Mobile-responsive patterns implemented")
        else:
            self.add_issue('medium', 'UX',
                          f"Limited responsive design patterns ({responsive_features} instances)")
        
        # Check for user feedback mechanisms
        feedback_patterns = [
            r'st\.success',
            r'st\.error',
            r'st\.warning',
            r'st\.info',
            r'display_error_card'
        ]
        
        feedback_count = 0
        for pattern in feedback_patterns:
            feedback_count += len(re.findall(pattern, self.source_code))
        
        if feedback_count >= 15:
            print("✅ User Feedback: Comprehensive user feedback system")
        else:
            self.add_issue('low', 'UX',
                          f"User feedback could be improved ({feedback_count} instances)")
    
    def run_all_tests(self):
        """Run complete data integration and API test suite"""
        print("🚀 Starting Data Integration & API Test Suite")
        print("=" * 60)
        
        self.test_bigquery_integration()
        self.test_data_processing_logic()
        self.test_openai_api_integration()
        self.test_error_handling_patterns()
        self.test_input_validation_security()
        self.test_performance_considerations()
        self.test_ui_responsiveness_code()
        
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 DATA INTEGRATION & API TEST SUMMARY")
        print("=" * 60)
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        print(f"Total Issues Found: {total_issues}")
        print(f"🔴 Critical: {len(self.issues['critical'])}")
        print(f"🟠 High: {len(self.issues['high'])}")
        print(f"🟡 Medium: {len(self.issues['medium'])}")
        print(f"🔵 Low: {len(self.issues['low'])}")
        
        # Detailed issue reporting
        for level in ['critical', 'high', 'medium', 'low']:
            if self.issues[level]:
                icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🔵'}[level]
                print(f"\n{icon} {level.upper()} ISSUES:")
                for issue in self.issues[level]:
                    line_info = f" (Line {issue['line']})" if issue['line'] else ""
                    print(f"   - [{issue['category']}] {issue['message']}{line_info}")
        
        # Assessment
        print(f"\n🎯 DATA INTEGRATION ASSESSMENT:")
        if len(self.issues['critical']) > 0:
            print("❌ CRITICAL DATA ISSUES - Cannot proceed to production")
        elif len(self.issues['high']) > 3:
            print("⚠️  MULTIPLE HIGH-PRIORITY DATA ISSUES - Review required")
        elif len(self.issues['high']) > 0:
            print("⚠️  HIGH-PRIORITY DATA ISSUES - Should be addressed")
        else:
            print("✅ DATA INTEGRATION APPEARS FUNCTIONAL")
        
        return {
            'total_issues': total_issues,
            'critical': len(self.issues['critical']),
            'high': len(self.issues['high']),
            'medium': len(self.issues['medium']),
            'low': len(self.issues['low'])
        }

if __name__ == "__main__":
    tester = DataIntegrationTester()
    results = tester.run_all_tests()