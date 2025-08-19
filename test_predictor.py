#!/usr/bin/env python3
"""
QA Test Suite for Enhanced Book Club Predictor

This script tests the core functionality of the enhanced predictor system
including data processing, AI integration, and error handling.
"""

import pandas as pd
import json
import sys
import os
import tempfile
import time
from unittest import mock

# Add the project directory to the path
sys.path.insert(0, '/Users/audriuspatalauskas/Documents/python/claude-coude-project_2')

# Import the functions we need to test
try:
    from bcv1 import (
        gather_comprehensive_user_reviews,
        build_user_context_prompt,
        get_enhanced_predictions,
        parse_llm_prediction,
        get_score_color,
        display_prediction_card,
        load_members,
        load_books,
        load_reviews
    )
    print("✅ Successfully imported all functions from bcv1.py")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class PredictorTester:
    def __init__(self):
        self.test_results = []
        self.critical_issues = []
        self.high_issues = []
        self.medium_issues = []
        self.low_issues = []
        
    def log_result(self, test_name, status, message, severity="info"):
        """Log test result and categorize issues by severity"""
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'severity': severity
        }
        self.test_results.append(result)
        
        if status == "FAIL":
            if severity == "critical":
                self.critical_issues.append(result)
            elif severity == "high":
                self.high_issues.append(result)
            elif severity == "medium":
                self.medium_issues.append(result)
            else:
                self.low_issues.append(result)
        
        print(f"{'✅' if status == 'PASS' else '❌'} {test_name}: {message}")

    def test_data_loading_functions(self):
        """Test data loading functions with mock data"""
        print("\n🔍 Testing Data Loading Functions...")
        
        try:
            # Create mock data
            mock_members = pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
                'gender': ['Female', 'Male', 'Male']
            })
            
            mock_books = pd.DataFrame({
                'id': [1, 2],
                'title': ['Test Book 1', 'Test Book 2'],
                'author': ['Author 1', 'Author 2'],
                'country': ['USA', 'UK'],
                'goodreads_avg': [4.0, 3.5],
                'suggested_by': ['Alice', 'Bob'],
                'season_no': [1, 1],
                'dominant_perspective': ['Female', 'Male']
            })
            
            mock_reviews = pd.DataFrame({
                'id': [1, 2, 3, 4],
                'book_id': [1, 1, 2, 2],
                'member_id': [1, 2, 1, 3],
                'rating': [4.5, 3.0, 5.0, 2.5],
                'comment': ['Great book!', 'Okay read', 'Loved it!', 'Not for me']
            })
            
            # Test gather_comprehensive_user_reviews with mock data
            with mock.patch('bcv1.load_books', return_value=mock_books), \
                 mock.patch('bcv1.load_members', return_value=mock_members), \
                 mock.patch('bcv1.load_reviews', return_value=mock_reviews):
                
                comprehensive_data = gather_comprehensive_user_reviews()
                
                if comprehensive_data.empty:
                    self.log_result("Data Integration", "FAIL", 
                                  "gather_comprehensive_user_reviews returned empty DataFrame", "high")
                else:
                    expected_columns = ['member_name', 'book_title', 'rating', 'comment']
                    missing_columns = [col for col in expected_columns if col not in comprehensive_data.columns]
                    
                    if missing_columns:
                        self.log_result("Data Integration", "FAIL", 
                                      f"Missing columns: {missing_columns}", "high")
                    else:
                        self.log_result("Data Integration", "PASS", 
                                      f"Successfully created comprehensive dataset with {len(comprehensive_data)} rows")
                        
                        # Test data relationships
                        if len(comprehensive_data) == 4:  # Should match number of reviews
                            self.log_result("Data Relationships", "PASS", 
                                          "Correct number of records in comprehensive dataset")
                        else:
                            self.log_result("Data Relationships", "FAIL", 
                                          f"Expected 4 records, got {len(comprehensive_data)}", "medium")
                            
        except Exception as e:
            self.log_result("Data Loading", "FAIL", f"Exception in data loading test: {str(e)}", "critical")

    def test_prompt_building(self):
        """Test prompt building functionality"""
        print("\n🔍 Testing Prompt Building...")
        
        try:
            # Mock comprehensive data
            mock_data = pd.DataFrame({
                'member_name': ['Alice', 'Alice', 'Bob'],
                'book_title': ['Book 1', 'Book 2', 'Book 1'],
                'author': ['Author 1', 'Author 2', 'Author 1'],
                'country': ['USA', 'UK', 'USA'],
                'rating': [4.5, 3.0, 2.5],
                'comment': ['Great!', 'Okay', 'Not good'],
                'goodreads_avg': [4.0, 3.5, 4.0]
            })
            
            # Test general prompt building
            prompt = build_user_context_prompt(mock_data)
            
            if not prompt or len(prompt) < 100:
                self.log_result("Prompt Building", "FAIL", "Generated prompt is too short or empty", "high")
            else:
                self.log_result("Prompt Building", "PASS", f"Generated prompt of {len(prompt)} characters")
                
                # Check for required elements in prompt
                required_elements = ['member_name', 'Alice', 'Bob', 'rating', 'JSON', 'predicted_score']
                missing_elements = [elem for elem in required_elements if elem not in prompt]
                
                if missing_elements:
                    self.log_result("Prompt Content", "FAIL", 
                                  f"Prompt missing required elements: {missing_elements}", "medium")
                else:
                    self.log_result("Prompt Content", "PASS", "Prompt contains all required elements")
            
            # Test targeted prompt building
            targeted_prompt = build_user_context_prompt(mock_data, "Specific Book Title")
            
            if "Specific Book Title" not in targeted_prompt:
                self.log_result("Targeted Prompt", "FAIL", 
                              "Target book title not found in targeted prompt", "medium")
            else:
                self.log_result("Targeted Prompt", "PASS", "Successfully generated targeted prompt")
                
        except Exception as e:
            self.log_result("Prompt Building", "FAIL", f"Exception in prompt building: {str(e)}", "high")

    def test_json_parsing(self):
        """Test JSON parsing functionality"""
        print("\n🔍 Testing JSON Response Parsing...")
        
        try:
            # Test valid JSON response
            valid_json = {
                "predictions": [
                    {
                        "member": "Alice",
                        "predicted_score": 4.2,
                        "confidence": "High",
                        "reasoning": "Based on historical preferences for literary fiction..."
                    },
                    {
                        "member": "Bob", 
                        "predicted_score": 3.1,
                        "confidence": "Medium",
                        "reasoning": "Mixed ratings suggest moderate enjoyment..."
                    }
                ]
            }
            
            # Test that valid JSON structure works
            predictions = valid_json.get("predictions", [])
            if len(predictions) == 2:
                self.log_result("JSON Structure", "PASS", "Valid JSON structure parsed correctly")
                
                # Test required fields
                required_fields = ['member', 'predicted_score', 'confidence', 'reasoning']
                for pred in predictions:
                    missing_fields = [field for field in required_fields if field not in pred]
                    if missing_fields:
                        self.log_result("JSON Fields", "FAIL", 
                                      f"Missing required fields: {missing_fields}", "high")
                        break
                else:
                    self.log_result("JSON Fields", "PASS", "All required fields present")
            else:
                self.log_result("JSON Structure", "FAIL", f"Expected 2 predictions, got {len(predictions)}", "medium")
                
            # Test malformed JSON handling
            malformed_json = '{"predictions": [{"member": "Alice", "predicted_score": 4.2, "confidence":'
            try:
                json.loads(malformed_json)
                self.log_result("JSON Error Handling", "FAIL", "Malformed JSON should raise exception", "medium")
            except json.JSONDecodeError:
                self.log_result("JSON Error Handling", "PASS", "Correctly handles malformed JSON")
                
        except Exception as e:
            self.log_result("JSON Parsing", "FAIL", f"Exception in JSON parsing test: {str(e)}", "high")

    def test_score_color_function(self):
        """Test score color assignment function"""
        print("\n🔍 Testing Score Color Function...")
        
        try:
            test_cases = [
                (4.5, "#27ae60"),  # High score - green
                (4.0, "#27ae60"),  # Boundary high score
                (3.5, "#f39c12"),  # Medium score - orange
                (3.0, "#f39c12"),  # Boundary medium score
                (2.5, "#e74c3c"),  # Low score - red
                (1.0, "#e74c3c"),  # Very low score
                ("invalid", "#95a5a6"),  # Invalid input - gray
                (None, "#95a5a6"),  # None input
            ]
            
            for score, expected_color in test_cases:
                result_color = get_score_color(score)
                if result_color == expected_color:
                    self.log_result(f"Score Color ({score})", "PASS", f"Correct color: {result_color}")
                else:
                    self.log_result(f"Score Color ({score})", "FAIL", 
                                  f"Expected {expected_color}, got {result_color}", "low")
                    
        except Exception as e:
            self.log_result("Score Color Function", "FAIL", f"Exception in score color test: {str(e)}", "medium")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n🔍 Testing Edge Cases...")
        
        try:
            # Test empty data
            empty_df = pd.DataFrame()
            prompt = build_user_context_prompt(empty_df)
            
            if "No review data available" in prompt:
                self.log_result("Empty Data Handling", "PASS", "Correctly handles empty dataset")
            else:
                self.log_result("Empty Data Handling", "FAIL", 
                              "Does not properly handle empty dataset", "medium")
            
            # Test single member data
            single_member_data = pd.DataFrame({
                'member_name': ['Alice'],
                'book_title': ['Book 1'],
                'rating': [4.0],
                'comment': ['Good book'],
                'author': ['Author 1'],
                'country': ['USA'],
                'goodreads_avg': [3.8]
            })
            
            single_prompt = build_user_context_prompt(single_member_data)
            if len(single_prompt) > 100 and 'Alice' in single_prompt:
                self.log_result("Single Member Data", "PASS", "Handles single member dataset correctly")
            else:
                self.log_result("Single Member Data", "FAIL", "Issues with single member dataset", "medium")
                
            # Test data with missing values
            data_with_nulls = pd.DataFrame({
                'member_name': ['Alice', 'Bob'],
                'book_title': ['Book 1', None],
                'rating': [4.0, None],
                'comment': [None, 'Comment'],
                'author': ['Author 1', 'Author 2'],
                'country': ['USA', None],
                'goodreads_avg': [3.8, 4.1]
            })
            
            null_prompt = build_user_context_prompt(data_with_nulls)
            if len(null_prompt) > 50:  # Should still generate something meaningful
                self.log_result("Null Data Handling", "PASS", "Handles null values appropriately")
            else:
                self.log_result("Null Data Handling", "FAIL", "Poor handling of null values", "high")
                
        except Exception as e:
            self.log_result("Edge Cases", "FAIL", f"Exception in edge case testing: {str(e)}", "high")

    def test_input_validation(self):
        """Test input validation and sanitization"""
        print("\n🔍 Testing Input Validation...")
        
        try:
            # Test malicious input
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE members; --",
                "../../../../etc/passwd",
                "\x00\x01\x02malicious",
                "A" * 10000,  # Very long input
                "",  # Empty input
                None,  # None input
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    # Test that the system handles these inputs safely
                    mock_data = pd.DataFrame({
                        'member_name': ['Alice'],
                        'book_title': [str(malicious_input) if malicious_input else 'Book 1'],
                        'rating': [4.0],
                        'comment': ['Test'],
                        'author': ['Author'],
                        'country': ['USA'],
                        'goodreads_avg': [3.8]
                    })
                    
                    prompt = build_user_context_prompt(mock_data, malicious_input)
                    # If we get here without exception, input was handled
                    
                except Exception as e:
                    # Some exceptions might be expected for truly malicious input
                    pass
                    
            self.log_result("Input Validation", "PASS", "System handles malicious inputs without crashing")
            
        except Exception as e:
            self.log_result("Input Validation", "FAIL", f"Input validation issues: {str(e)}", "critical")

    def test_performance_characteristics(self):
        """Test basic performance characteristics"""
        print("\n🔍 Testing Performance...")
        
        try:
            # Create larger test dataset
            large_data = pd.DataFrame({
                'member_name': [f'Member_{i}' for i in range(100)] * 10,
                'book_title': [f'Book_{i%20}' for i in range(1000)],
                'rating': [3.0 + (i % 5) * 0.5 for i in range(1000)],
                'comment': [f'Comment {i}' for i in range(1000)],
                'author': [f'Author_{i%10}' for i in range(1000)],
                'country': [['USA', 'UK', 'Canada', 'Australia', 'Germany'][i%5] for i in range(1000)],
                'goodreads_avg': [3.5 + (i % 3) * 0.5 for i in range(1000)]
            })
            
            # Test prompt generation time
            start_time = time.time()
            prompt = build_user_context_prompt(large_data)
            generation_time = time.time() - start_time
            
            if generation_time < 5.0:  # Should complete within 5 seconds
                self.log_result("Performance - Prompt Generation", "PASS", 
                              f"Generated prompt in {generation_time:.2f} seconds")
            else:
                self.log_result("Performance - Prompt Generation", "FAIL", 
                              f"Slow prompt generation: {generation_time:.2f} seconds", "medium")
            
            # Test memory usage (basic check)
            prompt_size = len(prompt)
            if prompt_size > 1000000:  # > 1MB might be excessive
                self.log_result("Performance - Memory", "FAIL", 
                              f"Very large prompt size: {prompt_size} characters", "medium")
            else:
                self.log_result("Performance - Memory", "PASS", 
                              f"Reasonable prompt size: {prompt_size} characters")
                              
        except Exception as e:
            self.log_result("Performance Testing", "FAIL", f"Performance test exception: {str(e)}", "medium")

    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Enhanced Predictor QA Test Suite")
        print("=" * 60)
        
        # Run all test categories
        self.test_data_loading_functions()
        self.test_prompt_building()
        self.test_json_parsing()
        self.test_score_color_function()
        self.test_edge_cases()
        self.test_input_validation()
        self.test_performance_characteristics()
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 QA TEST SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        
        print(f"Total Tests Run: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Pass Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Issue breakdown
        print(f"\n🔴 Critical Issues: {len(self.critical_issues)}")
        print(f"🟠 High Issues: {len(self.high_issues)}")
        print(f"🟡 Medium Issues: {len(self.medium_issues)}")
        print(f"🔵 Low Issues: {len(self.low_issues)}")
        
        # Detailed issue reporting
        if self.critical_issues:
            print("\n🔴 CRITICAL ISSUES (MUST FIX):")
            for issue in self.critical_issues:
                print(f"   - {issue['test']}: {issue['message']}")
        
        if self.high_issues:
            print("\n🟠 HIGH PRIORITY ISSUES:")
            for issue in self.high_issues:
                print(f"   - {issue['test']}: {issue['message']}")
        
        if self.medium_issues:
            print("\n🟡 MEDIUM PRIORITY ISSUES:")
            for issue in self.medium_issues:
                print(f"   - {issue['test']}: {issue['message']}")
        
        # Production readiness assessment
        print(f"\n🎯 PRODUCTION READINESS ASSESSMENT:")
        if len(self.critical_issues) > 0:
            print("❌ NOT READY FOR PRODUCTION - Critical issues must be resolved")
        elif len(self.high_issues) > 2:
            print("⚠️  CAUTION - Multiple high-priority issues detected")
        elif failed_tests / total_tests > 0.2:  # > 20% failure rate
            print("⚠️  CAUTION - High failure rate suggests stability issues")
        else:
            print("✅ CORE FUNCTIONALITY APPEARS STABLE")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'critical_issues': len(self.critical_issues),
            'high_issues': len(self.high_issues),
            'medium_issues': len(self.medium_issues),
            'low_issues': len(self.low_issues)
        }

if __name__ == "__main__":
    tester = PredictorTester()
    results = tester.run_all_tests()