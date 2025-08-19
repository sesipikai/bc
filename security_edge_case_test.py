#!/usr/bin/env python3
"""
Security and Edge Case Testing Suite

This script performs security vulnerability assessment and edge case testing
for the enhanced book club predictor system.
"""

import re
import json
import os

class SecurityEdgeCaseTester:
    def __init__(self):
        self.security_issues = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        self.edge_case_issues = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        # Read source code
        with open('/Users/audriuspatalauskas/Documents/python/claude-coude-project_2/bcv1.py', 'r') as f:
            self.source_code = f.read()
            self.lines = self.source_code.split('\n')
    
    def add_security_issue(self, level, message, line_num=None):
        """Add a security issue"""
        issue = {'message': message, 'line': line_num}
        self.security_issues[level].append(issue)
        print(f"🔒 {'❌' if level in ['critical', 'high'] else '⚠️' if level == 'medium' else 'ℹ️'} Security ({level}): {message}")
    
    def add_edge_case_issue(self, level, message, line_num=None):
        """Add an edge case issue"""
        issue = {'message': message, 'line': line_num}
        self.edge_case_issues[level].append(issue)
        print(f"🎯 {'❌' if level in ['critical', 'high'] else '⚠️' if level == 'medium' else 'ℹ️'} Edge Case ({level}): {message}")
    
    def test_authentication_security(self):
        """Test authentication security measures"""
        print("\n🔍 Testing Authentication Security...")
        
        # Check password hashing
        if 'bcrypt.checkpw' in self.source_code:
            print("✅ Password Security: Using bcrypt for password verification")
        else:
            self.add_security_issue('critical', "Password verification not using secure hashing")
        
        # Check for password exposure in logs/errors
        password_patterns = [
            r'print.*password',
            r'st\.write.*password',
            r'st\.error.*password',
            r'logging.*password'
        ]
        
        password_exposure = 0
        for i, line in enumerate(self.lines, 1):
            for pattern in password_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    password_exposure += 1
                    self.add_security_issue('high', f"Potential password exposure in output", i)
        
        if password_exposure == 0:
            print("✅ Password Exposure: No obvious password leakage in output")
        
        # Check session state security
        session_patterns = [
            r'st\.session_state\["authentication_status"\]',
            r'st\.session_state\["username"\]',
            r'st\.session_state\["role"\]'
        ]
        
        session_security = 0
        for pattern in session_patterns:
            if re.search(pattern, self.source_code):
                session_security += 1
        
        if session_security >= 3:
            print("✅ Session Management: Proper session state management")
        else:
            self.add_security_issue('medium', f"Incomplete session management ({session_security}/3 patterns)")
    
    def test_input_sanitization(self):
        """Test input sanitization and validation"""
        print("\n🔍 Testing Input Sanitization...")
        
        # Check for SQL injection prevention
        sql_injection_safe = True
        dangerous_sql_patterns = [
            r'f".*SELECT.*{[^}]+}"',  # Direct variable interpolation in SQL
            r'".*SELECT.*" \+ ',      # String concatenation in SQL
            r'\.format.*SELECT'       # Format method in SQL
        ]
        
        for i, line in enumerate(self.lines, 1):
            for pattern in dangerous_sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    sql_injection_safe = False
                    self.add_security_issue('critical', f"Potential SQL injection vulnerability", i)
        
        if sql_injection_safe:
            print("✅ SQL Injection: No obvious SQL injection vulnerabilities")
        
        # Check for XSS prevention
        xss_vulnerabilities = []
        for i, line in enumerate(self.lines, 1):
            # Check for unescaped user data in HTML
            if 'unsafe_allow_html=True' in line:
                # Look for user data variables in surrounding context
                context_start = max(0, i-5)
                context_end = min(len(self.lines), i+5)
                context = '\n'.join(self.lines[context_start:context_end])
                
                user_data_vars = ['member_name', 'book_title', 'comment', 'author', 'reasoning']
                for var in user_data_vars:
                    if f'{{{var}}}' in context:
                        xss_vulnerabilities.append((i, var))
        
        if len(xss_vulnerabilities) == 0:
            print("✅ XSS Prevention: No obvious XSS vulnerabilities detected")
        else:
            for line_num, var in xss_vulnerabilities:
                self.add_security_issue('high', f"Potential XSS vulnerability with variable {var}", line_num)
        
        # Check for CSRF protection (limited for Streamlit apps)
        if 'st.form' in self.source_code:
            print("✅ CSRF Protection: Using Streamlit forms for state management")
        else:
            self.add_security_issue('low', "Consider using st.form for better state management")
    
    def test_api_security(self):
        """Test API security measures"""
        print("\n🔍 Testing API Security...")
        
        # Check API key handling
        api_key_patterns = [
            r'st\.secrets\["openai_api_key"\]',
            r'api_key=.*st\.secrets',
            r'OpenAI\(api_key=.*\)'
        ]
        
        secure_api_handling = 0
        for pattern in api_key_patterns:
            if re.search(pattern, self.source_code):
                secure_api_handling += 1
        
        if secure_api_handling >= 2:
            print("✅ API Key Security: Secure API key handling")
        else:
            self.add_security_issue('high', f"Insecure API key handling ({secure_api_handling} secure patterns)")
        
        # Check for API key exposure
        api_exposure_patterns = [
            r'print.*api_key',
            r'st\.write.*api_key',
            r'logging.*api_key'
        ]
        
        api_exposure = False
        for i, line in enumerate(self.lines, 1):
            for pattern in api_exposure_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    api_exposure = True
                    self.add_security_issue('critical', f"API key exposure in logs/output", i)
        
        if not api_exposure:
            print("✅ API Key Exposure: No API key leakage detected")
        
        # Check rate limiting considerations
        if 'sleep(' in self.source_code or 'time.sleep' in self.source_code:
            print("✅ Rate Limiting: Some rate limiting measures implemented")
        else:
            self.add_security_issue('medium', "No obvious rate limiting for API calls")
    
    def test_data_validation_edge_cases(self):
        """Test edge cases in data validation"""
        print("\n🔍 Testing Data Validation Edge Cases...")
        
        # Check for empty data handling
        empty_data_patterns = [
            r'if.*\.empty:',
            r'if.*len\(.*\) == 0:',
            r'if not.*:.*return'
        ]
        
        empty_data_handling = 0
        for pattern in empty_data_patterns:
            empty_data_handling += len(re.findall(pattern, self.source_code))
        
        if empty_data_handling >= 5:
            print("✅ Empty Data Handling: Comprehensive empty data checks")
        else:
            self.add_edge_case_issue('medium', f"Limited empty data handling ({empty_data_handling} checks)")
        
        # Check for null/None handling
        null_patterns = [
            r'is None',
            r'== None',
            r'pd\.isna',
            r'pd\.notna',
            r'\.fillna',
            r'\.dropna'
        ]
        
        null_handling = 0
        for pattern in null_patterns:
            null_handling += len(re.findall(pattern, self.source_code))
        
        if null_handling >= 10:
            print("✅ Null Handling: Comprehensive null value processing")
        else:
            self.add_edge_case_issue('medium', f"Limited null handling ({null_handling} instances)")
        
        # Check for data type validation
        type_validation_patterns = [
            r'pd\.to_numeric.*errors=',
            r'isinstance\(',
            r'type\(.*\)',
            r'\.astype\('
        ]
        
        type_validation = 0
        for pattern in type_validation_patterns:
            type_validation += len(re.findall(pattern, self.source_code))
        
        if type_validation >= 5:
            print("✅ Type Validation: Good data type validation")
        else:
            self.add_edge_case_issue('medium', f"Limited type validation ({type_validation} instances)")
    
    def test_error_boundary_conditions(self):
        """Test error handling for boundary conditions"""
        print("\n🔍 Testing Error Boundary Conditions...")
        
        # Check for division by zero protection
        division_patterns = [
            r'/(?!\*)',  # Division operator not followed by *
            r'\.mean\(\)',
            r'sum\(.*\).*len\('
        ]
        
        potential_divisions = 0
        for pattern in division_patterns:
            potential_divisions += len(re.findall(pattern, self.source_code))
        
        # Check for zero checks
        zero_checks = len(re.findall(r'!= 0|== 0|> 0|< 0', self.source_code))
        
        if zero_checks >= potential_divisions // 2:
            print("✅ Division Safety: Adequate zero-division protection")
        else:
            self.add_edge_case_issue('medium', f"Potential division by zero risks ({zero_checks} checks vs {potential_divisions} operations)")
        
        # Check for array/list bounds checking
        indexing_patterns = [
            r'\[0\]',
            r'\[\d+\]',
            r'\.iloc\[',
            r'\.loc\['
        ]
        
        indexing_operations = 0
        for pattern in indexing_patterns:
            indexing_operations += len(re.findall(pattern, self.source_code))
        
        bounds_checks = len(re.findall(r'len\(.*\).*>', self.source_code))
        
        if bounds_checks > 0:
            print("✅ Bounds Checking: Some array bounds checking implemented")
        else:
            self.add_edge_case_issue('medium', f"Limited bounds checking ({bounds_checks} checks vs {indexing_operations} indexing operations)")
    
    def test_resource_limits(self):
        """Test resource usage and limits"""
        print("\n🔍 Testing Resource Limits...")
        
        # Check for large data handling
        large_data_patterns = [
            r'range\(\d{3,}\)',  # Large ranges
            r'for.*in.*range\(\d{4,}\)',  # Very large loops
            r'\.head\(',  # Data limiting
            r'\.sample\('  # Data sampling
        ]
        
        resource_management = 0
        large_operations = 0
        
        for pattern in large_data_patterns:
            matches = len(re.findall(pattern, self.source_code))
            if 'head(' in pattern or 'sample(' in pattern:
                resource_management += matches
            else:
                large_operations += matches
        
        if resource_management > 0:
            print("✅ Resource Management: Data limiting mechanisms present")
        else:
            self.add_edge_case_issue('medium', "No obvious data limiting for large datasets")
        
        if large_operations > 2:
            self.add_edge_case_issue('medium', f"Multiple large data operations ({large_operations}) may cause performance issues")
        
        # Check for memory usage patterns
        memory_intensive_patterns = [
            r'\.to_list\(\)',
            r'\.unique\(\)\.tolist\(\)',
            r'pd\.concat.*ignore_index=True'
        ]
        
        memory_operations = 0
        for pattern in memory_intensive_patterns:
            memory_operations += len(re.findall(pattern, self.source_code))
        
        if memory_operations > 10:
            self.add_edge_case_issue('medium', f"Many memory-intensive operations ({memory_operations}) - consider optimization")
        else:
            print(f"ℹ️ Memory Usage: {memory_operations} potentially memory-intensive operations")
    
    def test_concurrent_access_safety(self):
        """Test safety for concurrent access"""
        print("\n🔍 Testing Concurrent Access Safety...")
        
        # Check for global state modifications
        global_modifications = [
            r'st\.session_state\[.*\] =',
            r'global ',
            r'st\.cache.*clear'
        ]
        
        global_mod_count = 0
        for pattern in global_modifications:
            global_mod_count += len(re.findall(pattern, self.source_code))
        
        if global_mod_count <= 10:
            print("✅ Concurrent Safety: Limited global state modifications")
        else:
            self.add_edge_case_issue('low', f"Many global state modifications ({global_mod_count}) may cause concurrency issues")
        
        # Check for file system operations
        file_operations = [
            r'open\(',
            r'with open',
            r'os\.path',
            r'file\.'
        ]
        
        file_ops = 0
        for pattern in file_operations:
            file_ops += len(re.findall(pattern, self.source_code))
        
        if file_ops > 0:
            print(f"ℹ️ File Operations: {file_ops} file operations detected - ensure thread safety")
        else:
            print("✅ File Safety: No obvious file system operations")
    
    def run_all_tests(self):
        """Run complete security and edge case test suite"""
        print("🚀 Starting Security & Edge Case Test Suite")
        print("=" * 60)
        
        self.test_authentication_security()
        self.test_input_sanitization()
        self.test_api_security()
        self.test_data_validation_edge_cases()
        self.test_error_boundary_conditions()
        self.test_resource_limits()
        self.test_concurrent_access_safety()
        
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive security and edge case report"""
        print("\n" + "=" * 60)
        print("📊 SECURITY & EDGE CASE TEST SUMMARY")
        print("=" * 60)
        
        total_security = sum(len(issues) for issues in self.security_issues.values())
        total_edge_cases = sum(len(issues) for issues in self.edge_case_issues.values())
        total_issues = total_security + total_edge_cases
        
        print(f"Total Issues Found: {total_issues}")
        print(f"🔒 Security Issues: {total_security}")
        print(f"🎯 Edge Case Issues: {total_edge_cases}")
        
        print(f"\nSECURITY BREAKDOWN:")
        print(f"🔴 Critical: {len(self.security_issues['critical'])}")
        print(f"🟠 High: {len(self.security_issues['high'])}")
        print(f"🟡 Medium: {len(self.security_issues['medium'])}")
        print(f"🔵 Low: {len(self.security_issues['low'])}")
        
        print(f"\nEDGE CASE BREAKDOWN:")
        print(f"🔴 Critical: {len(self.edge_case_issues['critical'])}")
        print(f"🟠 High: {len(self.edge_case_issues['high'])}")
        print(f"🟡 Medium: {len(self.edge_case_issues['medium'])}")
        print(f"🔵 Low: {len(self.edge_case_issues['low'])}")
        
        # Report critical and high issues
        all_critical = self.security_issues['critical'] + self.edge_case_issues['critical']
        all_high = self.security_issues['high'] + self.edge_case_issues['high']
        
        if all_critical:
            print(f"\n🔴 CRITICAL ISSUES:")
            for issue in all_critical:
                line_info = f" (Line {issue['line']})" if issue['line'] else ""
                print(f"   - {issue['message']}{line_info}")
        
        if all_high:
            print(f"\n🟠 HIGH PRIORITY ISSUES:")
            for issue in all_high:
                line_info = f" (Line {issue['line']})" if issue['line'] else ""
                print(f"   - {issue['message']}{line_info}")
        
        # Final security assessment
        print(f"\n🎯 SECURITY ASSESSMENT:")
        if len(all_critical) > 0:
            print("❌ CRITICAL SECURITY ISSUES - Cannot deploy to production")
        elif len(all_high) > 2:
            print("⚠️  MULTIPLE HIGH-PRIORITY SECURITY ISSUES - Review required")
        elif len(all_high) > 0:
            print("⚠️  HIGH-PRIORITY SECURITY ISSUES - Should be addressed")
        elif total_security > 5:
            print("⚠️  SEVERAL SECURITY CONCERNS - Review recommended")
        else:
            print("✅ SECURITY POSTURE APPEARS ACCEPTABLE")
        
        return {
            'total_issues': total_issues,
            'security_critical': len(self.security_issues['critical']),
            'security_high': len(self.security_issues['high']),
            'security_medium': len(self.security_issues['medium']),
            'security_low': len(self.security_issues['low']),
            'edge_case_critical': len(self.edge_case_issues['critical']),
            'edge_case_high': len(self.edge_case_issues['high']),
            'edge_case_medium': len(self.edge_case_issues['medium']),
            'edge_case_low': len(self.edge_case_issues['low'])
        }

if __name__ == "__main__":
    tester = SecurityEdgeCaseTester()
    results = tester.run_all_tests()