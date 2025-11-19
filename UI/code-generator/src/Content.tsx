import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/GridLegacy';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import SearchIcon from '@mui/icons-material/Search';
import Box from "@mui/material/Box";
import {useState, useEffect, useCallback, useRef} from "react";
import axios from "axios";

export default function Content() {
    const [prompts, setPrompts] = useState<string[]>([]);
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [currentPrompt, setCurrentPrompt] = useState<string>("");
    const [results, setResults] = useState<string[]>([]);
    const [isProcessing, setIsProcessing] = useState<boolean>(false);
    const [autoStart, setAutoStart] = useState(false);
    const runIdRef = useRef(0);

    // Load questions on mount
    useEffect(() => {
        fetch("../question.json")
            .then(res => res.json())
            .then((data: { questions: string[] }) => {
                setPrompts(data.questions);
                if (data.questions.length > 0) {
                    setCurrentPrompt(data.questions[0]);
                }
            })
            .catch(err => console.error(err));
    }, []);

    const sendPrompt = useCallback(async () => {
        if (currentPrompt === "" || isProcessing) return;
        setIsProcessing(true);
        const thisRunId = runIdRef.current;
        try {
            const {data} = await axios.post(
                "http://172.17.0.1:4010/api/query",
                {prompt: currentPrompt},
                {headers: {"Content-Type": "application/json"}}
            );

            // Cancel outdated runs
            if (thisRunId !== runIdRef.current) return;

            // Append result
            setResults(prev => [...prev, data.result || JSON.stringify(data)]);

            // Move to next prompt
            setCurrentIndex(prevIndex => {
                if (thisRunId !== runIdRef.current) return prevIndex;
                const nextIndex = prevIndex + 1;
                if (nextIndex < prompts.length) {
                    setCurrentPrompt(prompts[nextIndex]);
                    return nextIndex;
                }

                // End of chain
                setAutoStart(false);
                setCurrentPrompt("");
                return prevIndex;
            });
        } catch (err) {
            console.error(err);
        } finally {
            // Cancel outdated runs
            if (thisRunId === runIdRef.current) {
                setIsProcessing(false);
            }
        }
    }, [currentPrompt, prompts, isProcessing]);

    const handleQueryClick = async () => {
        runIdRef.current += 1;   // cancel all old runs
        setResults([]);
        setCurrentIndex(0);
        setAutoStart(true);
        await sendPrompt();
    };

    // Auto-processing when prompt changes
    useEffect(() => {
        if (!autoStart || currentPrompt === "") return;
        (async () => {
            await sendPrompt();
        })();

    }, [currentPrompt, autoStart, sendPrompt]);

    return (
        <Paper sx={{maxWidth: 936, margin: 'auto', overflow: 'hidden'}}>
            <AppBar
                position="static"
                color="default"
                elevation={0}
                sx={{borderBottom: '1px solid rgba(0, 0, 0, 0.12)'}}
            >
                <Toolbar>
                    <Grid container spacing={2} sx={{alignItems: 'center'}}>
                        <Grid item>
                            <SearchIcon color="inherit" sx={{display: 'block'}}/>
                        </Grid>
                        <Grid item xs>
                            <TextField
                                fullWidth
                                placeholder="Post your questions here."
                                multiline={true}
                                value={currentPrompt
                                    ? `${currentPrompt} : (${currentIndex + 1}/${prompts.length})`
                                    : ""}
                                InputProps={{
                                    disableUnderline: true,
                                    sx: {fontSize: '15', color: 'green'},
                                }}
                                variant="standard"
                            />
                        </Grid>
                        <Grid item>
                            <Button variant="contained" sx={{mr: 1}} onClick={handleQueryClick} disabled={isProcessing}>
                                {isProcessing ? "Processing..." : "Query"}
                            </Button>
                        </Grid>
                    </Grid>
                </Toolbar>
            </AppBar>
            <Box sx={{maxHeight: 900, overflowY: "auto", p: 2, backgroundColor: "#e0e0e0"}}>
                {results.length === 0 ? (
                    <Typography align="center" sx={{color: "#4527a0", my: 5, fontSize: '16'}}>
                        No results yet
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