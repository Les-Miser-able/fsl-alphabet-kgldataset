// Vinext calls process.exit(0) while native Windows build handles are closing.
// Allow normal event-loop shutdown on success; preserve every failing exit.
const exit = process.exit.bind(process);
process.exit = function(code) {
 const status = code ?? process.exitCode ?? 0;
 if (Number(status) !== 0) return exit(status);
 process.exitCode = 0;
};
