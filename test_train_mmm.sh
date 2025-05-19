#!/bin/bash
# Run train_mmm.py with our minimal test configuration and data
export DATA_FILE="test_data.csv"
export CONFIG_FILE="test_mmm_config.json"
export OUTPUT_FILE="test_model_output.json"

# Run the enhanced train_mmm.py script
python python_scripts/train_mmm.py

# Check if the output file was created
echo "Checking if output file was created..."
if [ -f "$OUTPUT_FILE" ]; then
  echo "Output file created successfully: $OUTPUT_FILE"
  echo "Examining JSON structure..."
  
  # Check for the presence of key elements
  echo "Checking for channel_impact section..."
  cat "$OUTPUT_FILE" | grep -q '"channel_impact"' && echo "✓ channel_impact section found" || echo "× channel_impact section missing"
  
  echo "Checking for time_series_decomposition..."
  cat "$OUTPUT_FILE" | grep -q '"time_series_decomposition"' && echo "✓ time_series_decomposition found" || echo "× time_series_decomposition missing"
  
  echo "Checking for response_curves..."
  cat "$OUTPUT_FILE" | grep -q '"response_curves"' && echo "✓ response_curves found" || echo "× response_curves missing"
  
  echo "Checking for historical_spends..."
  cat "$OUTPUT_FILE" | grep -q '"historical_spends"' && echo "✓ historical_spends found" || echo "× historical_spends missing"
  
  echo "Checking for total_contributions_summary..."
  cat "$OUTPUT_FILE" | grep -q '"total_contributions_summary"' && echo "✓ total_contributions_summary found" || echo "× total_contributions_summary missing"
  
  # Extract and summarize the channel_impact section
  echo "Creating simplified channel_impact-only view..."
  cat "$OUTPUT_FILE" | jq '{ channel_impact }' > test_channel_impact_only.json
  echo "Created test_channel_impact_only.json for easier inspection"
else
  echo "Error: Output file not created. Check for errors in train_mmm.py execution."
fi