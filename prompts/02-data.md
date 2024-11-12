Create a responsive HTML mock for LearnSphere, a Learning Management System. The mock should include:
1. Navigation Bar
    * Logo
    * Main navigation items: Dashboard, Courses, Students, Analytics, Settings
    * User profile menu in top right
1. Main Dashboard View should include:
    * Welcome message
    * Quick stats cards showing:
    * Total active courses
    * Total enrolled students
    * Completion rate
    * Average assessment score
    * Recent activity feed
    * Course progress overview
1. Visual Style:
    * Use a professional, modern color scheme
    * Include appropriate icons for navigation and stats
    * Implement a clean, minimal design
    * Use Tailwind CSS for styling
    * Use placeholder images where needed
1. Requirements:
    * The page should be responsive
    * Use semantic HTML5 elements
    * Include appropriate hover states for interactive elements
    * Comments explaining the layout structure
    * Placeholder data that represents realistic usage
The mock should work as a standalone HTML file and use only CDN-hosted dependencies. Include appropriate padding and spacing between elements.


--- 

Create a modern, multi-step registration form for LearnSphere demo access. The form should Tailwind CSS, implementing a step-by-step wizard interface that collects all necessary information while maintaining a clean, user-friendly experience.

Form Requirements:

1. Step 1: Basic Information
- Company/Institution name
- Full name
- Work email
- Phone number (optional)
- How did you hear about us? (dropdown)
  * Google Search
  * Social Media
  * Education Conference
  * Product Review Site
  * Referral
  * Direct Traffic

2. Step 2: Organization Details
- Organization type (radio buttons)
  * Higher Education
  * Corporate Training
  * K-12 School
  * Professional Training Provider
  * Individual Course Creator
  * Certification Body
- Organization size (radio buttons)
  * 1-50 employees
  * 51-200 employees
  * 201-500 employees
  * 501-1000 employees
  * 1000+ employees
- Expected number of students (radio buttons)
  * <100
  * 100-500
  * 501-1000
  * 1000-5000
  * 5000+

3. Step 3: Role & Requirements
- Your role (dropdown)
  * Training Manager
  * Teacher/Professor
  * Course Creator
  * HR Manager
  * Department Head
  * IT Administrator
  * Independent Instructor
- Primary use case (radio buttons)
  * Employee Training
  * Academic Courses
  * Professional Certification
  * Compliance Training
  * Skills Development
  * Customer Training
- Required features (multi-select checkboxes)
  * Course Creation
  * Virtual Classroom
  * Assessment Tools
  * Mobile Learning
  * Analytics & Reporting
  * Integration Capabilities
  * Certification Management
  * Compliance Training
  * Content Library
- Current solution (radio buttons)
  * No LMS
  * Competitor Product
  * Custom Built Solution
  * Basic Tools
- Decision timeframe (dropdown)
  * Within 1 month
  * 1-3 months
  * 3-6 months
  * 6+ months
  * Just exploring

UI/UX Requirements:
1. Progress indicator showing current step
2. Previous/Next navigation buttons
3. Form validation with error messages
4. Save progress capability
5. Mobile-responsive design
6. Loading states for submissions
7. Success message after completion

Visual Requirements:
1. Use shadcn/ui components
2. Professional color scheme
3. Clear typography hierarchy
4. Proper spacing between elements
5. Smooth transitions between steps
6. Icon usage where appropriate

Technical Requirements:
1. Field validation
2. Error handling
3. Loading states
4. Accessible form controls
5. Proper HTML5 input types
6. Required field indicators

The form should look professional and trustworthy, as it's often the first interaction potential customers have with LearnSphere.

---


Generate two related datasets for LearnSphere, a Learning Management System that offers 30-day free demos to potential customers. We need to analyze user behavior during the demo period to predict which users are likely to convert to paying customers.

1. First Dataset: Demo Users (users.csv)
Required columns:
- user_id: Unique identifier (format: USR001)
- signup_date: Timestamp of demo signup
- lead_source: How they found LearnSphere (Google Search, Social Media, Education Conference, Product Review Site, Referral, Direct Traffic)
- organization_type: Type of organization (Higher Education, Corporate Training, K-12 School, Professional Training Provider, Individual Course Creator, Certification Body)
- organization_size: Number of employees (1-50, 51-200, 201-500, 501-1000, 1000+)
- expected_student_count: Expected number of students (<100, 100-500, 501-1000, 1000-5000, 5000+)
- role_title: User's role (Training Manager, Teacher/Professor, Course Creator, HR Manager, Department Head, IT Administrator, Independent Instructor)
- primary_use_case: Main purpose (Employee Training, Academic Courses, Professional Certification, Compliance Training, Skills Development, Customer Training)
- required_features: Comma-separated list of required features (Course Creation, Virtual Classroom, Assessment Tools, Mobile Learning, Analytics & Reporting, Integration Capabilities, Certification Management, Compliance Training, Content Library)
- current_solution: Current LMS solution (No LMS, Competitor Product, Custom Built Solution, Basic Tools)
- decision_timeframe: Expected decision timeline (Within 1 month, 1-3 months, 3-6 months, 6+ months, Just exploring)
- converted: Boolean indicating if they became a paying customer
- conversion_date: Date when they converted (if applicable)

2. Second Dataset: Activity Logs (logs.csv)
Required columns:
- timestamp: When the action occurred
- user_id: References user_id from first dataset
- action_category: Type of activity (course_creation, content_management, assessment, user_management, platform_exploration, integration, virtual_classroom, analytics, support)
- action_type: Specific action taken (e.g., create_course, upload_video, create_quiz, add_student, view_features, setup_zoom, etc.)
- duration_seconds: Time spent on the action

Requirements for the data:
1. Generate 100 user records spanning a 3-month period
2. Create ~2000 log entries showing realistic usage patterns
3. Include patterns that could indicate likelihood to convert:
   - Users who spend more time in core features
   - Users who contact sales
   - Users who add multiple courses and students
   - Users who test integrations
4. Include realistic timestamps that make sense (during business hours, proper sequence)
5. Maintain realistic conversion rate (~30-40%)
6. Include both power users and casual explorers
7. Some users should have very few actions (abandoned demos)

The data should be suitable for training a lead scoring model to predict which demo users are likely to convert to paying customers.