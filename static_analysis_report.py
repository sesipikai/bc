#!/usr/bin/env python3
"""
Static Code Analysis Report for Enhanced Book Club Predictor

This script performs comprehensive static analysis of the bcv1.py code
to identify potential issues, security vulnerabilities, and code quality concerns.
"""

import re
import os
import json

class StaticAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.issues = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'info': []
        }
        
        # Read the source code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.source_code = f.read()
                self.lines = self.source_code.split('\n')
        except Exception as e:
            print(f"Error reading file: {e}")
            self.source_code = ""
            self.lines = []
    
    def add_issue(self, level, category, line_num, message, recommendation=""):
        """Add an issue to the analysis report"""
        issue = {
            'category': category,
            'line': line_num,
            'message': message,
            'recommendation': recommendation
        }
        self.issues[level].append(issue)
    
    def analyze_imports(self):
        """Analyze import statements for security and best practices"""
        print("🔍 Analyzing imports...")
        
        import_lines = [i for i, line in enumerate(self.lines, 1) 
                       if line.strip().startswith('import ') or line.strip().startswith('from ')]
        
        # Check for duplicate imports
        imports = []
        for i, line_content in enumerate(self.lines):
            if line_content.strip().startswith('import ') or line_content.strip().startswith('from '):
                imports.append((i+1, line_content.strip()))
        
        # Check for duplicate bigquery import
        bigquery_imports = [imp for imp in imports if 'bigquery' in imp[1]]
        if len(bigquery_imports) > 1:
            self.add_issue('medium', 'Code Quality', bigquery_imports[1][0],
                          "Duplicate BigQuery import detected",
                          "Remove duplicate import on line 12")
        
        # Check for security-sensitive imports
        sensitive_imports = ['subprocess', 'eval', 'exec', 'pickle']
        for line_num, import_line in imports:
            for sensitive in sensitive_imports:
                if sensitive in import_line:
                    self.add_issue('high', 'Security', line_num,
                                  f"Potentially dangerous import: {sensitive}",
                                  "Review usage for security implications")
    
    def analyze_secrets_usage(self):
        """Analyze secrets and credential handling"""
        print("🔍 Analyzing secrets handling...")
        
        secret_patterns = [
            (r'st\.secrets\[', 'Streamlit secrets access'),
            (r'api_key', 'API key reference'),
            (r'password', 'Password reference'),
            (r'credentials', 'Credentials reference')
        ]
        
        for i, line in enumerate(self.lines, 1):
            for pattern, description in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if it's in a try-except block (good practice)
                    context_lines = self.lines[max(0, i-10):i+5]
                    context = '\n'.join(context_lines)
                    
                    if 'try:' in context and 'except' in context:
                        self.add_issue('info', 'Security', i,
                                      f"{description} properly wrapped in try-except")
                    else:
                        self.add_issue('medium', 'Security', i,
                                      f"{description} not in error handling block",
                                      "Wrap secrets access in try-except for better error handling")
    
    def analyze_sql_injection_risks(self):
        """Check for SQL injection vulnerabilities"""
        print("🔍 Analyzing SQL injection risks...")
        
        sql_patterns = [
            r'f".*SELECT.*FROM.*{.*}"',
            r"f'.*SELECT.*FROM.*{.*}'",
            r'\.format\(.*SELECT.*FROM',
            r'query.*\+.*',
        ]
        
        for i, line in enumerate(self.lines, 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.add_issue('high', 'Security', i,
                                  "Potential SQL injection risk detected",
                                  "Use parameterized queries instead of string formatting")
        
        # Check BigQuery queries - these appear to be properly formatted
        bigquery_queries = []
        for i, line in enumerate(self.lines, 1):
            if 'SELECT * FROM' in line and '`{' in line:
                bigquery_queries.append((i, line.strip()))
        
        for line_num, query in bigquery_queries:
            if 'TABLE' in query:  # Using table constants
                self.add_issue('info', 'Security', line_num,
                              "BigQuery query using table constants (good practice)")
    
    def analyze_error_handling(self):
        """Analyze error handling patterns"""
        print("🔍 Analyzing error handling...")
        
        try_blocks = []
        for i, line in enumerate(self.lines, 1):
            if line.strip().startswith('try:'):
                # Find corresponding except block
                except_found = False
                for j in range(i, min(i+20, len(self.lines))):
                    if self.lines[j].strip().startswith('except'):
                        except_found = True
                        break
                
                if not except_found:
                    self.add_issue('high', 'Error Handling', i,
                                  "try block without corresponding except",
                                  "Add proper exception handling")
                else:
                    try_blocks.append(i)
        
        # Check for bare except clauses
        for i, line in enumerate(self.lines, 1):
            if re.match(r'\s*except:\s*$', line):
                self.add_issue('medium', 'Error Handling', i,
                              "Bare except clause catches all exceptions",
                              "Specify exception types for better error handling")
            elif 'except Exception as e:' in line:
                self.add_issue('info', 'Error Handling', i,
                              "Good practice: catching Exception with variable")
    
    def analyze_function_complexity(self):
        """Analyze function complexity and length"""
        print("🔍 Analyzing function complexity...")
        
        functions = []
        current_function = None
        current_function_line = 0
        indent_level = 0
        
        for i, line in enumerate(self.lines, 1):
            stripped = line.strip()
            if stripped.startswith('def '):
                if current_function:
                    # Previous function ended
                    length = i - current_function_line
                    functions.append((current_function, current_function_line, length))
                
                current_function = stripped.split('(')[0].replace('def ', '')
                current_function_line = i
                indent_level = len(line) - len(line.lstrip())
        
        # Add the last function
        if current_function:
            length = len(self.lines) - current_function_line
            functions.append((current_function, current_function_line, length))
        
        # Analyze function lengths
        for func_name, start_line, length in functions:
            if length > 100:
                self.add_issue('medium', 'Code Quality', start_line,
                              f"Function '{func_name}' is very long ({length} lines)",
                              "Consider breaking into smaller functions")
            elif length > 50:
                self.add_issue('low', 'Code Quality', start_line,
                              f"Function '{func_name}' is moderately long ({length} lines)")
        
        # Check for specific complex functions
        complex_functions = ['predictor', 'view_charts', 'get_enhanced_predictions']
        for func_name, start_line, length in functions:
            if func_name in complex_functions and length > 30:
                self.add_issue('info', 'Code Quality', start_line,
                              f"Complex function '{func_name}' ({length} lines) - review for refactoring opportunities")
    
    def analyze_openai_integration(self):
        """Analyze OpenAI integration for best practices"""
        print("🔍 Analyzing OpenAI integration...")
        
        openai_patterns = [
            (r'openai\.', 'OpenAI client usage'),
            (r'gpt-4o-mini', 'GPT model specification'),
            (r'response_format.*json_object', 'JSON response format'),
            (r'temperature.*=', 'Temperature setting'),
            (r'max_tokens.*=', 'Token limit setting')
        ]
        
        openai_usage_found = False
        for i, line in enumerate(self.lines, 1):
            for pattern, description in openai_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    openai_usage_found = True
                    self.add_issue('info', 'API Integration', i, f"{description} found")
        
        if openai_usage_found:
            # Check for proper error handling around OpenAI calls
            openai_call_lines = []
            for i, line in enumerate(self.lines, 1):
                if 'client.chat.completions.create' in line:
                    openai_call_lines.append(i)
            
            for call_line in openai_call_lines:
                # Check if it's in a try-except block
                context_start = max(0, call_line - 10)
                context_end = min(len(self.lines), call_line + 10)
                context = self.lines[context_start:context_end]
                
                try_found = any('try:' in line for line in context[:10])
                except_found = any('except' in line for line in context[10:])
                
                if try_found and except_found:
                    self.add_issue('info', 'API Integration', call_line,
                                  "OpenAI API call properly wrapped in error handling")
                else:
                    self.add_issue('medium', 'API Integration', call_line,
                                  "OpenAI API call should be in try-except block",
                                  "Wrap API calls in proper error handling")
    
    def analyze_data_validation(self):
        """Analyze data validation and sanitization"""
        print("🔍 Analyzing data validation...")
        
        # Check for input validation
        validation_patterns = [
            (r'\.strip\(\)', 'String stripping'),
            (r'pd\.to_numeric.*errors=', 'Pandas numeric conversion with error handling'),
            (r'if.*empty', 'Empty data check'),
            (r'\.dropna\(', 'Null value handling'),
            (r'errors="coerce"', 'Error coercion in data conversion')
        ]
        
        for i, line in enumerate(self.lines, 1):
            for pattern, description in validation_patterns:
                if re.search(pattern, line):
                    self.add_issue('info', 'Data Validation', i, f"Good practice: {description}")
        
        # Check for potential XSS vulnerabilities in HTML generation
        html_patterns = [
            r'unsafe_allow_html=True',
            r'st\.markdown.*<.*>',
            r'f".*<.*{.*}.*"'
        ]
        
        for i, line in enumerate(self.lines, 1):
            for pattern in html_patterns:
                if re.search(pattern, line):
                    # Check if user input might be involved
                    if 'member_name' in line or 'book_title' in line or 'comment' in line:
                        self.add_issue('high', 'Security', i,
                                      "Potential XSS vulnerability in HTML generation",
                                      "Sanitize user input before inserting into HTML")
                    else:
                        self.add_issue('low', 'Security', i,
                                      "HTML generation with unsafe_allow_html - review for XSS risks")
    
    def analyze_performance_concerns(self):
        """Analyze potential performance issues"""
        print("🔍 Analyzing performance concerns...")
        
        # Check for potential N+1 queries or inefficient operations
        inefficient_patterns = [
            (r'for.*in.*df\.iterrows\(\)', 'DataFrame iteration (potentially slow)'),
            (r'\.append\(.*\)', 'List/DataFrame append in loop (inefficient)'),
            (r'pd\.concat.*ignore_index=True', 'DataFrame concatenation (check if in loop)')
        ]
        
        for i, line in enumerate(self.lines, 1):
            for pattern, description in inefficient_patterns:
                if re.search(pattern, line):
                    self.add_issue('medium', 'Performance', i, description,
                                  "Consider more efficient alternatives")
        
        # Check for large data operations
        large_data_patterns = [
            (r'range\(1000', 'Large range operation'),
            (r'\.unique\(\)\.tolist\(\)', 'Potentially memory-intensive operation')
        ]
        
        for i, line in enumerate(self.lines, 1):
            for pattern, description in large_data_patterns:
                if re.search(pattern, line):
                    self.add_issue('low', 'Performance', i, description)
    
    def analyze_code_structure(self):
        """Analyze overall code structure and organization"""
        print("🔍 Analyzing code structure...")
        
        # Check file length
        total_lines = len(self.lines)
        if total_lines > 2000:
            self.add_issue('medium', 'Code Structure', 1,
                          f"File is very large ({total_lines} lines)",
                          "Consider splitting into multiple modules")
        elif total_lines > 1500:
            self.add_issue('low', 'Code Structure', 1,
                          f"File is large ({total_lines} lines)")
        
        # Check for magic numbers
        magic_numbers = re.findall(r'\b(?<![\w.])[0-9]{2,}\b(?![\w.])', self.source_code)
        unique_magic_numbers = set(magic_numbers)
        if len(unique_magic_numbers) > 10:
            self.add_issue('low', 'Code Quality', 1,
                          f"Many magic numbers found: {len(unique_magic_numbers)}",
                          "Consider using named constants")
        
        # Check for long lines
        long_lines = []
        for i, line in enumerate(self.lines, 1):
            if len(line) > 120:
                long_lines.append(i)
        
        if len(long_lines) > 10:
            self.add_issue('low', 'Code Quality', long_lines[0],
                          f"{len(long_lines)} lines exceed 120 characters",
                          "Consider line breaking for better readability")
    
    def run_analysis(self):
        """Run complete static analysis"""
        print("🔍 Starting Static Code Analysis...")
        print("=" * 50)
        
        if not self.source_code:
            print("❌ Could not read source code file")
            return
        
        # Run all analysis modules
        self.analyze_imports()
        self.analyze_secrets_usage()
        self.analyze_sql_injection_risks()
        self.analyze_error_handling()
        self.analyze_function_complexity()
        self.analyze_openai_integration()
        self.analyze_data_validation()
        self.analyze_performance_concerns()
        self.analyze_code_structure()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 50)
        print("📊 STATIC ANALYSIS REPORT")
        print("=" * 50)
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        print(f"Total Issues Found: {total_issues}")
        print(f"🔴 Critical: {len(self.issues['critical'])}")
        print(f"🟠 High: {len(self.issues['high'])}")
        print(f"🟡 Medium: {len(self.issues['medium'])}")
        print(f"🔵 Low: {len(self.issues['low'])}")
        print(f"ℹ️  Info: {len(self.issues['info'])}")
        
        # Detailed issue reporting
        for level in ['critical', 'high', 'medium', 'low']:
            if self.issues[level]:
                print(f"\n{level.upper()} ISSUES:")
                for issue in self.issues[level]:
                    print(f"   Line {issue['line']}: [{issue['category']}] {issue['message']}")
                    if issue['recommendation']:
                        print(f"      → Recommendation: {issue['recommendation']}")
        
        # Positive findings
        if self.issues['info']:
            print(f"\nℹ️  POSITIVE FINDINGS:")
            for issue in self.issues['info']:
                print(f"   Line {issue['line']}: [{issue['category']}] {issue['message']}")
        
        # Overall assessment
        print(f"\n🎯 STATIC ANALYSIS ASSESSMENT:")
        if len(self.issues['critical']) > 0:
            print("❌ CRITICAL ISSUES FOUND - Must be resolved before production")
        elif len(self.issues['high']) > 3:
            print("⚠️  MULTIPLE HIGH-PRIORITY ISSUES - Review recommended")
        elif len(self.issues['high']) > 0:
            print("⚠️  HIGH-PRIORITY ISSUES DETECTED - Should be addressed")
        elif len(self.issues['medium']) > 5:
            print("⚠️  SEVERAL MEDIUM-PRIORITY ISSUES - Consider addressing")
        else:
            print("✅ CODE QUALITY APPEARS ACCEPTABLE")
        
        return {
            'total_issues': total_issues,
            'critical': len(self.issues['critical']),
            'high': len(self.issues['high']),
            'medium': len(self.issues['medium']),
            'low': len(self.issues['low']),
            'info': len(self.issues['info']),
            'issues': self.issues
        }

if __name__ == "__main__":
    file_path = "/Users/audriuspatalauskas/Documents/python/claude-coude-project_2/bcv1.py"
    analyzer = StaticAnalyzer(file_path)
    results = analyzer.run_analysis()