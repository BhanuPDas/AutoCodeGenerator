
import './App.css'
import * as React from 'react';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormControl from '@mui/material/FormControl';
import FormLabel from '@mui/material/FormLabel';

export default function App() {
    const [value, setValue] = React.useState('female');

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setValue((event.target as HTMLInputElement).value);
};

  return (
      <div>
      <div className="d-block p-2 text-bg-primary">Code Generator!</div>
    <div>
        <FormControl>
            <FormLabel id="code-functions">Code Options</FormLabel>
            <RadioGroup
                row
                aria-labelledby="code-functions"
                name="code-radio-options"
                value={value}
                onChange={handleChange}
            >
                <FormControlLabel value="generate" control={<Radio />} label="Generate" />
                <FormControlLabel value="refactor" control={<Radio />} label="Refactor" />
                <FormControlLabel value="fixbug" control={<Radio />} label="Fix Bug" />
            </RadioGroup>
        </FormControl>
      </div>
      </div>
  )
}