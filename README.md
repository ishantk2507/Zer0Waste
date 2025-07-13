# ZeroWaste.AI WebApp

A retail food-waste and green logistics dashboard built with Flask, Tailwind CSS, and Plotly.js.

## Features

- **Retailer Login**: Simple authentication system
- **Inventory Check**: AI-powered freshness analysis with environmental data
- **Dashboard**: Real-time metrics, high-risk items tracking, and redistribution mapping
- **Responsive Design**: Mobile-first layout with Tailwind CSS
- **Interactive Visualizations**: Plotly.js maps and charts
- **Accessibility**: ARIA labels, keyboard navigation, and screen reader support

## Tech Stack

- **Backend**: Flask + Jinja2 templates
- **Frontend**: Tailwind CSS (CDN), Vanilla JavaScript
- **Visualizations**: Plotly.js
- **Styling**: Mobile-first responsive design

## Installation

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the application:
\`\`\`bash
python app.py
\`\`\`

3. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Login**: Use any username and password (demo mode)
2. **Inventory Check**: Enter product details to get freshness analysis
3. **Dashboard**: Monitor metrics, view high-risk items, and manage redistributions

## Routes

- `/` - Redirects to login
- `/login` - Authentication page
- `/inventory` - Freshness checking form
- `/predict` - API endpoint for freshness prediction
- `/dashboard` - Main dashboard with metrics and visualizations
- `/redistribute` - API endpoint for redistribution scheduling
- `/logout` - Clear session and redirect to login

## Features

### Inventory Management
- Product freshness scoring (0-100%)
- Environmental factor analysis (temperature, humidity)
- Storage type optimization
- Redistribution recommendations

### Dashboard Analytics
- Total units saved tracking
- CO₂ emission prevention metrics
- Green score ranking system
- High-risk items monitoring

### Redistribution System
- Interactive outlet selection
- Distance and capacity information
- SMS/email notification simulation
- Route optimization mapping

## Accessibility

- Semantic HTML structure
- ARIA labels and roles
- Keyboard navigation support
- Focus management
- Screen reader compatibility
- Color contrast compliance

## Mobile Responsiveness

- Mobile-first design approach
- Collapsible navigation
- Touch-friendly interactions
- Responsive grid layouts
- Optimized form inputs
