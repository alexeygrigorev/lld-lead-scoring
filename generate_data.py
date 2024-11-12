import pandas as pd
import numpy as np

from datetime import datetime, timedelta
import random

class DemoDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Date range
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 3, 31)
        
        # Define conversion likelihood modifiers for different attributes
        self.conversion_factors = {
            'lead_source': {
                'Education Conference': 0.4,
                'Product Review Site': 0.3,
                'Referral': 0.25,
                'Google Search': 0.2,
                'Social Media': 0.15,
                'Direct Traffic': 0.1
            },
            'organization_type': {
                'Corporate Training': 0.35,
                'Higher Education': 0.3,
                'Professional Training Provider': 0.3,
                'Certification Body': 0.25,
                'K-12 School': 0.2,
                'Individual Course Creator': 0.15
            },
            'organization_size': {
                '501-1000': 0.35,
                '1000+': 0.35,
                '201-500': 0.3,
                '51-200': 0.25,
                '1-50': 0.15
            },
            'decision_timeframe': {
                'Within 1 month': 0.4,
                '1-3 months': 0.3,
                '3-6 months': 0.2,
                '6+ months': 0.1,
                'Just exploring': 0.05
            }
        }
        
        # Activity patterns that indicate higher conversion likelihood
        self.high_value_actions = {
            'setup_integration': 0.8,
            'contact_sales': 0.7,
            'create_course': 0.6,
            'invite_users': 0.6,
            'configure_branding': 0.5,
            'create_assessment': 0.5
        }

    def generate_user_profile(self, user_id):
        """Generate a user profile with attributes that influence conversion probability"""
        
        # Select attributes based on realistic distributions
        lead_source = np.random.choice(
            list(self.conversion_factors['lead_source'].keys()),
            p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
        )
        
        org_type = np.random.choice(
            list(self.conversion_factors['organization_type'].keys()),
            p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.1]
        )
        
        org_size = np.random.choice(
            list(self.conversion_factors['organization_size'].keys()),
            p=[0.1, 0.2, 0.3, 0.2, 0.2]
        )
        
        decision_timeframe = np.random.choice(
            list(self.conversion_factors['decision_timeframe'].keys()),
            p=[0.2, 0.3, 0.25, 0.15, 0.1]
        )
        
        # Calculate base conversion probability
        conversion_prob = (
            self.conversion_factors['lead_source'][lead_source] +
            self.conversion_factors['organization_type'][org_type] +
            self.conversion_factors['organization_size'][org_size] +
            self.conversion_factors['decision_timeframe'][decision_timeframe]
        ) / 4
        
        # Add some randomness
        conversion_prob = min(0.9, max(0.1, conversion_prob + np.random.normal(0, 0.1)))
        
        features = [
            'Course Creation',
            'Virtual Classroom',
            'Assessment Tools',
            'Mobile Learning',
            'Analytics & Reporting',
            'Integration Capabilities',
            'Certification Management',
            'Compliance Training',
            'Content Library'
        ]
        
        # More interested users select more features
        num_features = int(np.random.normal(4 + conversion_prob * 4, 1))
        num_features = max(2, min(len(features), num_features))
        
        return {
            'user_id': user_id,
            'signup_date': self.start_date + timedelta(days=np.random.randint(0, 90)),
            'lead_source': lead_source,
            'organization_type': org_type,
            'organization_size': org_size,
            'expected_student_count': np.random.choice(
                ['<100', '100-500', '501-1000', '1000-5000', '5000+'],
                p=[0.2, 0.3, 0.25, 0.15, 0.1]
            ),
            'role_title': np.random.choice([
                'Training Manager',
                'Teacher/Professor',
                'Course Creator',
                'HR Manager',
                'Department Head',
                'IT Administrator',
                'Independent Instructor'
            ]),
            'primary_use_case': np.random.choice([
                'Employee Training',
                'Academic Courses',
                'Professional Certification',
                'Compliance Training',
                'Skills Development',
                'Customer Training'
            ]),
            'required_features': ', '.join(np.random.choice(features, num_features, replace=False)),
            'current_solution': np.random.choice([
                'No LMS',
                'Competitor Product',
                'Custom Built Solution',
                'Basic Tools'
            ]),
            'decision_timeframe': decision_timeframe,
            'base_conversion_prob': conversion_prob
        }

    def generate_user_activities(self, user_profile):
        """Generate activity logs based on user profile and likelihood to convert"""
        
        activities = []
        conversion_prob = user_profile['base_conversion_prob']
        signup_date = user_profile['signup_date']
        
        # Determine number of activities based on conversion probability
        num_activities = int(np.random.normal(20 + conversion_prob * 80, 10))
        num_activities = max(5, min(200, num_activities))
        
        # Basic action templates
        action_templates = {
            'course_creation': ['create_course', 'edit_course', 'publish_course'],
            'content_management': ['upload_video', 'create_document', 'organize_content'],
            'assessment': ['create_quiz', 'create_assignment', 'grade_submission'],
            'user_management': ['invite_users', 'create_group', 'assign_roles'],
            'platform_exploration': ['view_features', 'view_pricing', 'view_docs'],
            'integration': ['setup_integration', 'test_integration', 'configure_api'],
            'virtual_classroom': ['schedule_session', 'host_meeting', 'record_session'],
            'analytics': ['view_reports', 'export_data', 'configure_dashboard'],
            'support': ['contact_sales', 'open_ticket', 'view_help']
        }
        
        # Generate activities over the 30-day period
        for _ in range(num_activities):
            # Higher conversion probability leads to more high-value actions
            if np.random.random() < conversion_prob:
                category = np.random.choice(list(action_templates.keys()))
                action = np.random.choice(action_templates[category])
            else:
                category = 'platform_exploration'
                action = 'view_features'
            
            # Generate timestamp within 30 days of signup
            days_offset = np.random.beta(2, 5) * 30  # Front-load activities
            hours = np.random.normal(14, 3)  # Center around 2 PM
            hours = max(8, min(18, hours))  # Constrain to business hours
            
            timestamp = signup_date + timedelta(days=days_offset, hours=hours)
            
            # Duration based on action type and user engagement
            base_duration = np.random.normal(300, 100)  # 5 minutes average
            duration = int(max(30, base_duration * (0.5 + conversion_prob)))
            
            activities.append({
                'timestamp': timestamp,
                'user_id': user_profile['user_id'],
                'action_category': category,
                'action_type': action,
                'duration_seconds': duration
            })
        
        return activities

    def generate_datasets(self):
        """Generate both users and activities datasets"""
        users_data = []
        activities_data = []
        
        # Generate user profiles
        for i in range(100):
            user_id = f'USR{i+1:03d}'
            user_profile = self.generate_user_profile(user_id)
            
            # Generate activities for this user
            user_activities = self.generate_user_activities(user_profile)
            activities_data.extend(user_activities)
            
            # Determine conversion based on activities and base probability
            high_value_actions = sum(1 for activity in user_activities 
                                   if activity['action_type'] in self.high_value_actions)
            activity_score = min(1.0, high_value_actions / 10)
            
            final_conversion_prob = (user_profile['base_conversion_prob'] + activity_score) / 2
            converted = np.random.random() < final_conversion_prob
            
            # Add conversion data to user profile
            user_profile['converted'] = converted
            if converted:
                last_activity = max(activity['timestamp'] for activity in user_activities)
                conversion_date = last_activity + timedelta(days=np.random.randint(1, 5))
                user_profile['conversion_date'] = conversion_date
            else:
                user_profile['conversion_date'] = None
                
            del user_profile['base_conversion_prob']  # Remove temporary field
            users_data.append(user_profile)
        
        # Convert to DataFrames
        users_df = pd.DataFrame(users_data)
        activities_df = pd.DataFrame(activities_data)
        
        # Sort activities by timestamp
        activities_df = activities_df.sort_values('timestamp')
        
        return users_df, activities_df

# Generate the datasets
generator = DemoDataGenerator()
users_df, logs_df = generator.generate_datasets()

# Save to CSV files
users_df.to_csv('users.csv', index=False)
logs_df.to_csv('logs.csv', index=False)

# Print some statistics
print("\nDataset Statistics:")
print(f"Total users: {len(users_df)}")
print(f"Conversion rate: {(users_df['converted'].mean() * 100):.1f}%")
print(f"Total activities: {len(logs_df)}")
print(f"Average activities per user: {len(logs_df) / len(users_df):.1f}")