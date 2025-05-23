# UI Test Guide for Ridge Regression MMM

## Quick Test Workflow

### 1. Start Fresh
- Open browser to http://localhost:3000
- Create new project: "Ridge MMM Test"

### 2. Upload Data
- Upload file: `uploads/dankztestdata_v2.csv`
- Should see "90 rows uploaded successfully"

### 3. Column Mapping
- Date: `Date`
- Target: `Sales`
- Channels: Select all 5 spend columns
- Control: `interestrate_control`

### 4. Model Setup
- Name: "Test Ridge Model"
- Leave default adstock/saturation values
- Create Model

### 5. Monitor Training
- Watch progress bar
- Check browser console for errors (F12)
- Should complete in ~30 seconds

### 6. Results Verification

#### Overview Tab
- [ ] R-squared shows ~7.7%
- [ ] Top channel: PPCBrand_Spend
- [ ] ROI: ~$73.77

#### Sales Decomposition Tab
- [ ] Pie chart displays
- [ ] Shows ~77% base, ~23% marketing
- [ ] All channel names visible

#### Channel Impact Tab
- [ ] ROI bar chart shows PPCBrand highest
- [ ] Performance table populated
- [ ] Efficiency matrix displays

#### Media Mix Curves Tab
- [ ] Curves display (NOT "No data available")
- [ ] Shows L, k, x0 parameters
- [ ] Current spend positions marked

## Console Errors to Watch For
- "Cannot read property 'channels' of undefined"
- "No saturation curve data available"
- Any 500 errors from API calls