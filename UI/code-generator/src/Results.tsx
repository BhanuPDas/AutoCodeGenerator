import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Box from "@mui/material/Box";
import {useState, useEffect, useRef} from "react";
import axios from "axios";

export default function Results() {
    const [results, setResults] = useState<string[]>([]);
    const hasFetchedResults = useRef(false);

    useEffect(() => {
        if (hasFetchedResults.current) return;
        hasFetchedResults.current = true;

        const fetchResults = async () => {
            try {
                const {data} = await axios.get(
                    "http://172.22.174.173:4010/api/results",
                    {headers: {"Content-Type": "application/json"}}
                );
                setResults([data.result || JSON.stringify(data)]);
            } catch (err) {
                console.error(err);
            }
        };
        void fetchResults();
    }, []);

    return (
        <Paper sx={{maxWidth: 1024, margin: 'auto', overflow: 'hidden'}}>
            <Box sx={{maxHeight: 1024, overflowY: "auto", p: 2, backgroundColor: "#e0e0e0"}}>
                {!results || results.length === 0 ? (
                    <Typography align="center" sx={{color: "#4527a0", my: 5, fontSize: '16'}}>
                        No results yet. Please try again after sometime.
                    </Typography>
                ) : (
                    results.map((res, idx) => (
                        <Typography
                            key={idx}
                            sx={{
                                borderRadius: 1,
                                p: 1,
                                my: 1,
                                whiteSpace: "pre-wrap",
                                fontFamily: "monospace",
                                fontSize: "16",
                                color: "black"
                            }}
                        >
                            {res}
                        </Typography>
                    ))
                )}
            </Box>
        </Paper>
    );
}