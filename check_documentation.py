#!/usr/bin/env python3
"""
Script to check documentation status of Python files in the CT reconstruction project.

This script scans Python files and checks for:
- Presence of module docstrings
- Type hints in function definitions
- Class docstrings
- Function/method docstrings
"""

import os
import ast
import glob
from typing import List, Dict, Tuple

def has_module_docstring(tree: ast.AST) -> bool:
    """Check if module has a docstring."""
    return (isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Constant) and 
            isinstance(tree.body[0].value.value, str))

def has_type_hints(node: ast.FunctionDef) -> bool:
    """Check if function has type hints."""
    has_return_annotation = node.returns is not None
    has_arg_annotations = any(arg.annotation is not None for arg in node.args.args)
    return has_return_annotation or has_arg_annotations

def analyze_file(file_path: str) -> Dict[str, any]:
    """Analyze a Python file for documentation completeness."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=file_path)
        
        result = {
            'file': file_path,
            'module_docstring': False,
            'classes': [],
            'functions': [],
            'total_classes': 0,
            'documented_classes': 0,
            'total_functions': 0,
            'functions_with_docstrings': 0,
            'functions_with_type_hints': 0,
        }
        
        # Check module docstring
        if tree.body and has_module_docstring(tree):
            result['module_docstring'] = True
        
        # Analyze classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                result['total_classes'] += 1
                has_docstring = (node.body and 
                               isinstance(node.body[0], ast.Expr) and
                               isinstance(node.body[0].value, ast.Constant))
                if has_docstring:
                    result['documented_classes'] += 1
                result['classes'].append({
                    'name': node.name,
                    'has_docstring': has_docstring
                })
            
            elif isinstance(node, ast.FunctionDef):
                result['total_functions'] += 1
                has_docstring = (node.body and 
                               isinstance(node.body[0], ast.Expr) and
                               isinstance(node.body[0].value, ast.Constant))
                has_types = has_type_hints(node)
                
                if has_docstring:
                    result['functions_with_docstrings'] += 1
                if has_types:
                    result['functions_with_type_hints'] += 1
                    
                result['functions'].append({
                    'name': node.name,
                    'has_docstring': has_docstring,
                    'has_type_hints': has_types
                })
        
        return result
        
    except Exception as e:
        return {'file': file_path, 'error': str(e)}

def main():
    """Main function to analyze all Python files in the project."""
    base_path = "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/notebooks/__code"
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to analyze\n")
    
    total_stats = {
        'files': 0,
        'module_docstrings': 0,
        'total_classes': 0,
        'documented_classes': 0,
        'total_functions': 0,
        'functions_with_docstrings': 0,
        'functions_with_type_hints': 0,
    }
    
    well_documented = []
    needs_work = []
    
    for file_path in sorted(python_files):
        result = analyze_file(file_path)
        
        if 'error' in result:
            print(f"ERROR analyzing {result['file']}: {result['error']}")
            continue
            
        total_stats['files'] += 1
        if result['module_docstring']:
            total_stats['module_docstrings'] += 1
        total_stats['total_classes'] += result['total_classes']
        total_stats['documented_classes'] += result['documented_classes']
        total_stats['total_functions'] += result['total_functions']
        total_stats['functions_with_docstrings'] += result['functions_with_docstrings']
        total_stats['functions_with_type_hints'] += result['functions_with_type_hints']
        
        # Determine if file is well documented
        rel_path = os.path.relpath(file_path, base_path)
        
        doc_score = 0
        total_score = 0
        
        # Module docstring (weight: 1)
        total_score += 1
        if result['module_docstring']:
            doc_score += 1
            
        # Class documentation (weight: 2 per class)
        if result['total_classes'] > 0:
            total_score += result['total_classes'] * 2
            doc_score += result['documented_classes'] * 2
            
        # Function documentation (weight: 1 per function)
        if result['total_functions'] > 0:
            total_score += result['total_functions'] * 2  # docstring + type hints
            doc_score += result['functions_with_docstrings']
            doc_score += result['functions_with_type_hints']
        
        if total_score > 0:
            score_pct = (doc_score / total_score) * 100
            if score_pct >= 80:
                well_documented.append((rel_path, score_pct))
            else:
                needs_work.append((rel_path, score_pct, result))
    
    # Print summary
    print("=" * 60)
    print("DOCUMENTATION ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"Total files analyzed: {total_stats['files']}")
    print(f"Module docstrings: {total_stats['module_docstrings']}/{total_stats['files']} "
          f"({total_stats['module_docstrings']/max(total_stats['files'],1)*100:.1f}%)")
    
    if total_stats['total_classes'] > 0:
        print(f"Class docstrings: {total_stats['documented_classes']}/{total_stats['total_classes']} "
              f"({total_stats['documented_classes']/total_stats['total_classes']*100:.1f}%)")
    
    if total_stats['total_functions'] > 0:
        print(f"Function docstrings: {total_stats['functions_with_docstrings']}/{total_stats['total_functions']} "
              f"({total_stats['functions_with_docstrings']/total_stats['total_functions']*100:.1f}%)")
        print(f"Function type hints: {total_stats['functions_with_type_hints']}/{total_stats['total_functions']} "
              f"({total_stats['functions_with_type_hints']/total_stats['total_functions']*100:.1f}%)")
    
    print(f"\nWell documented files (≥80%): {len(well_documented)}")
    for file_path, score in well_documented:
        print(f"  ✓ {file_path} ({score:.1f}%)")
    
    print(f"\nFiles needing work (<80%): {len(needs_work)}")
    for file_path, score, details in needs_work[:10]:  # Show first 10
        print(f"  ✗ {file_path} ({score:.1f}%)")
        if details['total_functions'] > 0:
            print(f"    Functions: {details['functions_with_docstrings']}/{details['total_functions']} documented, "
                  f"{details['functions_with_type_hints']}/{details['total_functions']} typed")

if __name__ == "__main__":
    main()
